from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── LOAD MODEL AND MAPPINGS ───────────────────────────────────────────────────
model = joblib.load("models/gb_credit_model.pkl")

with open("models/risk_mappings.json", "r") as f:
    risk_mappings = json.load(f)

FEATURES = [
    'years_in_operation', 'num_employees', 'owner_age',
    'annual_revenue_ghs', 'monthly_momo_volume_ghs',
    'avg_monthly_bank_balance_ghs', 'bank_account_tenure_months',
    'has_momo_account', 'loan_amount_requested_ghs', 'collateral_value_ghs',
    'previous_loan_count', 'previous_default', 'credit_bureau_score',
    'is_female', 'has_valid_tin', 'has_collateral',
    'collateral_coverage_ratio', 'revenue_to_loan_ratio',
    'momo_activity_score', 'account_stability', 'is_established',
    'has_credit_score', 'loan_to_revenue_ratio', 'momo_to_loan_ratio',
    'balance_to_loan_ratio', 'credit_score_band', 'high_risk_flag',
    'revenue_per_employee', 'sector_risk_score', 'region_risk_score',
    'purpose_risk_score', 'age_risk'
]


def preprocess(app_data):
    df = pd.DataFrame([app_data])

    gender_map = {
        'male': 'Male', 'M': 'Male', 'm': 'Male',
        'female': 'Female', 'F': 'Female', 'f': 'Female'
    }
    df['owner_gender'] = df['owner_gender'].replace(gender_map)
    df['is_female'] = (df['owner_gender'] == 'Female').astype(int)

    def clean_currency(val):
        if pd.isna(val):
            return np.nan
        val = str(val).replace('GHS', '').replace('$', '').replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return np.nan

    df['annual_revenue_ghs'] = df['annual_revenue_ghs'].apply(clean_currency)

    bool_map = {
        'yes': 1, 'Yes': 1, 'YES': 1, 'TRUE': 1, 'True': 1,
        'true': 1, '1': 1, 1: 1, 'Y': 1, 'y': 1,
        'no': 0, 'No': 0, 'NO': 0, 'FALSE': 0, 'False': 0,
        'false': 0, '0': 0, 0: 0, 'N': 0, 'n': 0
    }
    df['has_momo_account'] = df['has_momo_account'].map(bool_map).fillna(0).astype(int)
    df['previous_default'] = df['previous_default'].map(bool_map).fillna(0).astype(int)

    df['has_valid_tin'] = (
        ~df['gra_tin'].isin(['PENDING', '', None]) &
        df['gra_tin'].notna()
    ).astype(int)

    df['collateral_type'] = df['collateral_type'].fillna('None')
    df['collateral_value_ghs'] = df['collateral_value_ghs'].fillna(0)
    df['has_collateral'] = (
        (df['collateral_type'] != 'None') &
        (df['collateral_value_ghs'] > 0)
    ).astype(int)
    df['collateral_coverage_ratio'] = np.where(
        df['loan_amount_requested_ghs'] > 0,
        df['collateral_value_ghs'] / df['loan_amount_requested_ghs'], 0
    )

    df['revenue_to_loan_ratio'] = np.where(
        df['loan_amount_requested_ghs'] > 0,
        df['annual_revenue_ghs'].fillna(0) / df['loan_amount_requested_ghs'], np.nan
    )
    df['momo_activity_score'] = np.where(
        df['has_momo_account'] == 1,
        df['monthly_momo_volume_ghs'].fillna(0), 0
    )
    df['account_stability'] = (
        df['avg_monthly_bank_balance_ghs'].fillna(0) *
        df['bank_account_tenure_months'].fillna(0)
    )
    df['is_established'] = (df['years_in_operation'] >= 2).astype(int)
    df['has_credit_score'] = df['credit_bureau_score'].notna().astype(int)

    train_medians = risk_mappings.get('train_medians', {})
    numeric_cols = [
        'annual_revenue_ghs', 'monthly_momo_volume_ghs',
        'avg_monthly_bank_balance_ghs', 'credit_bureau_score',
        'collateral_value_ghs', 'revenue_to_loan_ratio'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(train_medians.get(col, 0))

    df['loan_to_revenue_ratio'] = np.where(
        df['annual_revenue_ghs'] > 0,
        df['loan_amount_requested_ghs'] / df['annual_revenue_ghs'], 99
    )
    df['momo_to_loan_ratio'] = np.where(
        df['loan_amount_requested_ghs'] > 0,
        df['monthly_momo_volume_ghs'].fillna(0) * 12 / df['loan_amount_requested_ghs'], 0
    )
    df['balance_to_loan_ratio'] = np.where(
        df['loan_amount_requested_ghs'] > 0,
        df['avg_monthly_bank_balance_ghs'].fillna(0) / df['loan_amount_requested_ghs'], 0
    )
    df['credit_score_band'] = pd.cut(
        df['credit_bureau_score'].fillna(0),
        bins=[0, 400, 500, 600, 700, 900],
        labels=[1, 2, 3, 4, 5]
    ).astype(float)
    df['high_risk_flag'] = (
        (df['previous_default'] == 1) |
        (df['previous_loan_count'] >= 3)
    ).astype(int)
    df['revenue_per_employee'] = np.where(
        df['num_employees'] > 0,
        df['annual_revenue_ghs'].fillna(0) / df['num_employees'], 0
    )

    global_rate = risk_mappings.get('global_default_rate', 0.1327)
    df['sector_risk_score'] = df['sector'].map(
        risk_mappings.get('sector_rates', {})).fillna(global_rate)
    df['region_risk_score'] = df['region'].map(
        risk_mappings.get('region_rates', {})).fillna(global_rate)
    df['purpose_risk_score'] = df['loan_purpose'].map(
        risk_mappings.get('purpose_rates', {})).fillna(global_rate)
    df['age_risk'] = np.where(
        (df['owner_age'] < 25) | (df['owner_age'] > 65), 1, 0
    )

    return df


def decision_engine(score, app_data):
    if app_data.get('previous_default') == 1:
        return {'decision': 'DECLINE',
                'reason': 'HR001: Previous loan default on record',
                'score': round(score, 4), 'override': True}
    if app_data.get('days_past_due_current', 0) > 90:
        return {'decision': 'DECLINE',
                'reason': 'HR002: Currently 90+ days past due',
                'score': round(score, 4), 'override': True}
    revenue = app_data.get('annual_revenue_ghs', 0) or 0
    loan = app_data.get('loan_amount_requested_ghs', 0) or 0
    if revenue > 0 and loan > (5 * revenue):
        return {'decision': 'REFER TO HUMAN',
                'reason': 'HR003: Loan exceeds 5x annual revenue',
                'score': round(score, 4), 'override': True}
    if app_data.get('years_in_operation', 0) < 0.5 and app_data.get('has_collateral', 0) == 0:
        return {'decision': 'DECLINE',
                'reason': 'HR004: Business under 6 months with no collateral',
                'score': round(score, 4), 'override': True}
    if score < 0.25:
        return {'decision': 'APPROVE',
                'reason': f'Score {score:.4f} below approval threshold (0.25)',
                'score': round(score, 4), 'override': False}
    elif score > 0.55:
        return {'decision': 'DECLINE',
                'reason': f'Score {score:.4f} exceeds decline threshold (0.55)',
                'score': round(score, 4), 'override': False}
    else:
        return {'decision': 'REFER TO HUMAN',
                'reason': f'Score {score:.4f} in review zone (0.25-0.55)',
                'score': round(score, 4), 'override': False}


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Stanbic Bank Ghana — SME Credit Assessment API',
        'version': '1.0',
        'model': risk_mappings.get('best_model_name', 'Logistic Regression'),
        'auc': risk_mappings.get('best_auc', 0.64),
        'status': 'live',
        'endpoints': {
            'health': 'GET /',
            'score': 'POST /score'
        }
    })


@app.route('/score', methods=['POST'])
def score():
    try:
        application = request.get_json()

        df = preprocess(application)
        train_medians = risk_mappings.get('train_medians', {})
        features = df[FEATURES].fillna(pd.Series(train_medians))

        prob = float(model.predict_proba(features)[0, 1])
        result = decision_engine(prob, df.iloc[0].to_dict())

        return jsonify({
            'application_id':      application.get('application_id', 'N/A'),
            'business_name':       application.get('business_name', 'N/A'),
            'default_probability': result['score'],
            'decision':            result['decision'],
            'reason':              result['reason'],
            'override':            result['override'],
            'status':              'success'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
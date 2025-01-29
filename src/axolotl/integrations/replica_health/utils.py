from io import StringIO


def text_to_df(text: str):
    import pandas as pd
    df = pd.read_csv(StringIO(text), names=["blood_glucose","insulin_infusion","date","description","heartrate","ice"])
    return df

def text_to_bg_preds(text: str):
    df = text_to_df(text)
    return df.blood_glucose

def text_to_hr_preds(text: str):
    df = text_to_df(text)
    return df.heartrate
    
def text_to_insulin_preds(text: str):
    df = text_to_df(text)
    return df.insulin_infusion



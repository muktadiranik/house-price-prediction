import joblib
import numpy as np

columns = {
    "columns": [
        "Garage_NO",
        "Garage_YES",
        "Porch_NO",
        "Porch_YES",
        "PoolQC_Ex",
        "PoolQC_Fa",
        "PoolQC_Gd",
        "PoolQC_NA",
        "Electrical_FuseA",
        "Electrical_FuseF",
        "Electrical_FuseP",
        "Electrical_Mix",
        "Electrical_SBrkr",
        "CentralAir_N",
        "CentralAir_Y",
        "Heating_Floor",
        "Heating_GasA",
        "Heating_GasW",
        "Heating_Grav",
        "Heating_OthW",
        "Heating_Wall",
        "ExterCond_Ex",
        "ExterCond_Fa",
        "ExterCond_Gd",
        "ExterCond_Po",
        "ExterCond_TA",
        "RoofStyle_Flat",
        "RoofStyle_Gable",
        "RoofStyle_Gambrel",
        "RoofStyle_Hip",
        "RoofStyle_Mansard",
        "RoofStyle_Shed",
        "HouseStyle_1.5Fin",
        "HouseStyle_1.5Unf",
        "HouseStyle_1Story",
        "HouseStyle_2.5Fin",
        "HouseStyle_2.5Unf",
        "HouseStyle_2Story",
        "HouseStyle_SFoyer",
        "HouseStyle_SLvl",
        "BldgType_1Fam",
        "BldgType_2fmCon",
        "BldgType_Duplex",
        "BldgType_Twnhs",
        "BldgType_TwnhsE",
        "Neighborhood_Blmngtn",
        "Neighborhood_Blueste",
        "Neighborhood_BrDale",
        "Neighborhood_BrkSide",
        "Neighborhood_ClearCr",
        "Neighborhood_CollgCr",
        "Neighborhood_Crawfor",
        "Neighborhood_Edwards",
        "Neighborhood_Gilbert",
        "Neighborhood_IDOTRR",
        "Neighborhood_MeadowV",
        "Neighborhood_Mitchel",
        "Neighborhood_NAmes",
        "Neighborhood_NPkVill",
        "Neighborhood_NWAmes",
        "Neighborhood_NoRidge",
        "Neighborhood_NridgHt",
        "Neighborhood_OldTown",
        "Neighborhood_SWISU",
        "Neighborhood_Sawyer",
        "Neighborhood_SawyerW",
        "Neighborhood_Somerst",
        "Neighborhood_StoneBr",
        "Neighborhood_Timber",
        "Neighborhood_Veenker",
        "Utilities_AllPub",
        "Utilities_ELO",
        "Utilities_NoSeWa",
        "Alley_Grvl",
        "Alley_NA",
        "Alley_Pave",
        "Street_Grvl",
        "Street_Pave",
        "MSZoning_C (all)",
        "MSZoning_FV",
        "MSZoning_RH",
        "MSZoning_RL",
        "MSZoning_RM",
        "LotArea",
        "TotalBsmtSF",
        "GrLivArea",
        "FullBath",
        "BedroomAbvGr",
        "KitchenAbvGr"
    ]
}

columns = columns["columns"]


def Garage():
    return columns[0:2]


def Porch():
    return columns[2:4]


def CentralAir():
    return columns[13:15]


def Utilities():
    return columns[70:73]


def Alley():
    return columns[73:76]


def Street():
    return columns[76:78]


def MSZoning():
    return columns[78:83]


def Neighborhood():
    return columns[45:70]


def BldgType():
    return columns[40:45]


def HouseStyle():
    return columns[32:40]


def RoofStyle():
    return columns[26:32]


def ExterCond():
    return columns[21:26]


def Heating():
    return columns[15:21]


def Electrical():
    return columns[8:13]


def PoolQC():
    return columns[4:8]


def predict_price(_Garage, _Porch, _PoolQC, _Electrical, _CentralAir, _Heating, _ExterCond, _RoofStyle, _HouseStyle, _BldgType, _Neighborhood, _Utilities, _Alley, _Street, _MSZoning, _LotArea, _TotalBsmtSF, _GrLivArea, _FullBath, _BedroomAbvGr, _KitchenAbvGr):
    with open('./artifacts/house_price_prediction_1.jl', 'rb') as f:
        __model = joblib.load(f)
        X = np.zeros(len(columns))
        Garage_index = columns.index(_Garage)
        if Garage_index > 0:
            X[Garage_index] = 1
        Porch_index = columns.index(_Porch)
        if Porch_index > 0:
            X[Porch_index] = 1
        PoolQC_index = columns.index(_PoolQC)
        if PoolQC_index > 0:
            X[PoolQC_index] = 1
        Electrical_index = columns.index(_Electrical)
        if Electrical_index >= 0:
            X[Electrical_index] = 1
        CentralAir_index = columns.index(_CentralAir)
        if CentralAir_index >= 0:
            X[CentralAir_index] = 1
        Heating_index = columns.index(_Heating)
        if Heating_index >= 0:
            X[Heating_index] = 1
        ExterCond_index = columns.index(_ExterCond)
        if ExterCond_index >= 0:
            X[ExterCond_index] = 1
        RoofStyle_index = columns.index(_RoofStyle)
        if RoofStyle_index >= 0:
            X[RoofStyle_index] = 1
        HouseStyle_index = columns.index(_HouseStyle)
        if HouseStyle_index >= 0:
            X[HouseStyle_index] = 1
        BldgType_index = columns.index(_BldgType)
        if BldgType_index >= 0:
            X[BldgType_index] = 1
        Neighborhood_index = columns.index(_Neighborhood)
        if Neighborhood_index >= 0:
            X[Neighborhood_index] = 1
        Utilities_index = columns.index(_Utilities)
        if Utilities_index >= 0:
            X[Utilities_index] = 1
        Alley_index = columns.index(_Alley)
        if Alley_index >= 0:
            X[Alley_index] = 1
        Street_index = columns.index(_Street)
        if Street_index >= 0:
            X[Street_index] = 1
        MSZoning_index = columns.index(_MSZoning)
        if MSZoning_index >= 0:
            X[MSZoning_index] = 1
        X[83] = _LotArea
        X[84] = _TotalBsmtSF
        X[85] = _GrLivArea
        X[86] = _FullBath
        X[87] = _BedroomAbvGr
        X[88] = _KitchenAbvGr
        return round(__model.predict([X])[0], 2)

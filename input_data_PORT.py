input_data = [
    {
        "species": ["Si"],
        "model": "SW_StillingerWeber_1985_Si__MO_405512056662_006",
        "model_shortname": "StillingerWebber"
    },
    {
        "species": ["Si"],
        "model": "SW_LeeHwang_2012GGA_Si__MO_040570764911_001",
        "model_shortname": "SW_LeeHwang"
    },
    {
        "species": ["Al"],
        "model": "EAM_CubicNaturalSpline_ErcolessiAdams_1994_Al__MO_800509458712_003",
        "model_shortname": "EAM_ErcolessiAdams"
    },
    {
        "species": ["Ag"],
        "model": "EAM_Dynamo_AcklandTichyVitek_1987v2_Ag__MO_055919219575_001",
        "model_shortname": "EAM_AcklandTichyVitek"
    },
    {
        "species": ["Fe", "C"],
        "model": "EAM_Dynamo_AlleraRibeiroPerez_2022_FeC__MO_324606345076_000",
        "model_shortname": "EAM_AlleraRibeiroPerez"
    },
    {
        "species": ["Pd", "Ag", "H"],
        "model": "EAM_Dynamo_HaleWongZimmerman_2008PairHybrid_PdAgH__MO_104806802344_006",
        "model_shortname": "EAM_HaleWongZimmerman"
    },
    {
        "species": ["Al", "Ni", "Co"],
        "model": "EAM_IMD_BrommerGaehler_2006A_AlNiCo__MO_122703700223_003",
        "model_shortname": "EAM_BrommerGaehler"
    },
    {
        "species": ["Ca", "Cd"],
        "model": "EAM_IMD_BrommerGaehlerMihalkovic_2007_CaCd__MO_145183423516_003",
        "model_shortname": "EAM_BrommerGaehlerMihalkovic"
    },
    {
        "species": ["Fe"],
        "model": "EAM_Magnetic2GQuintic_ChiesaDerletDudarev_2011_Fe__MO_140444321607_002",
        "model_shortname": "EAM_ChiesaDerletDudarev"
    },
    {
        "species": ["V"],
        "model": "EAM_MagneticCubic_DerletNguyenDudarev_2007_V__MO_683890323730_002",
        "model_shortname": "EAM_DerletNguyenDudarev"
    },
    {
        "species": ["Fe"],
        "model": "EAM_MagneticCubic_MendelevHanSrolovitz_2003_Fe__MO_856295952425_002",
        "model_shortname": "EAM_MendelevHanSrolovitz"
    },
    {
        "species": ["Ge"],
        "model": "EDIP_BelkoGusakovDorozhkin_2010_Ge__MO_129433059219_001",
        "model_shortname": "EDIP_BelkoGusakovDorozhkin"
    },
    {
        "species": ["Ag"],
        "model": "EMT_Asap_Standard_JacobsenStoltzeNorskov_1996_Ag__MO_303974873468_001",
        "model_shortname": "EMT_JacobsenStoltzeNorskov"
    },
    {
        "species": ["Si","C"],
        "model": "LJ_ElliottAkerson_2015_Universal__MO_959249795837_003",
        "model_shortname": "LJ_ElliottAkerson"
    },
    {
        "species": ["Ar"],
        "model": "LJ_Shifted_Bernardes_1958LowCutoff_Ar__MO_720819638419_004",
        "model_shortname": "LJ_Bernardes"
    },
    {
        "species": ["Ar"],
        "model": "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004",
        "model_shortname": "LJ_Bernardes"
    },
    {
        "species": ["Ar"],
        "model": "LJ_Truncated_Nguyen_2005_Ar__MO_398194508715_001",
        "model_shortname": "LJ_Nguyen"
    },
    {
        "species": ["Si"],
        "model": "MEAM_LAMMPS_HuangDongLiu_2018_Si__MO_050147023220_002",
        "model_shortname": "MEAM_HuangDongLiu"
    },
    {
        "species": ["Si", "C"],
        "model": "MEAM_LAMMPS_KangEunJun_2014_SiC__MO_477506997611_002",
        "model_shortname": "MEAM_KangEunJun"
    },
    {
        "species": ["Si"],
        "model": "MEAM_LAMMPS_Lee_2007_Si__MO_774917820956_001",
        "model_shortname": "MEAM_Lee"
    },
    {
        "species": ["Si"],
        "model": "MFF_MistriotisFlytzanisFarantos_1989_Si__MO_080526771943_001",
        "model_shortname": "MFF_MistriotisFlytzanisFarantos"
    },
    {
        "species": ["Ar"],
        "model": "Morse_QuinticSmoothed_Jelinek_1972_Ar__MO_908645784389_002",
        "model_shortname": "Morse_Jelinek"
    },
    {
        "species": ["Ag"],
        "model": "Morse_Shifted_GirifalcoWeizer_1959HighCutoff_Ag__MO_111986436268_004",
        "model_shortname": "Morse_GirifalcoWeizer"
    },
    {
        "species": ["Fe"],
        "model": "Morse_Shifted_GirifalcoWeizer_1959HighCutoff_Fe__MO_147603128437_004",
        "model_shortname": "Morse_GirifalcoWeizer"
    },
    {
        "species": ["Ag"],
        "model": "Morse_Shifted_GirifalcoWeizer_1959LowCutoff_Ag__MO_137719994600_004",
        "model_shortname": "Morse_GirifalcoWeizer"
    },
    {
        "species": ["Ag"],
        "model": "Morse_Shifted_GirifalcoWeizer_1959MedCutoff_Ag__MO_861893969202_004",
        "model_shortname": "Morse_GirifalcoWeizer"
    },
    {
        "species": ["Ar"],
        "model": "Morse_Shifted_Jelinek_1972_Ar__MO_831902330215_004",
        "model_shortname": "Morse_Jelinek"
    },
    {
        "species": ["Nb", "Ta", "W", "Mo"],
        "model": "SNAP_LiChenZheng_2019_NbTaWMo__MO_560387080449_000",
        "model_shortname": "SNAP_LiChenZheng"
    },
    {
        "species": ["Cd", "Te"],
        "model": "SW_WangStroudMarkworth_1989_CdTe__MO_786496821446_001",
        "model_shortname": "SW_WangStroudMarkworth"
    },

]


import numpy as np

pos_weight = np.array(
    [
        0.85715023,
        0.76151323,
        0.70327359,
        1.57061349,
        0.60561147,
        0.74534094,
        1.44137769,
        1.34570526,
        0.57182919,
        0.38269696,
        1.75589024,
        1.13234485,
        0.61947749,
        0.94978794,
        0.50926186,
        0.66613861,
        1.35191577,
        1.64414869,
        0.93914605,
        0.66026402,
        0.69467116,
        1.00903369,
        0.78238945,
        0.45192687,
        1.09732181,
        1.34570526,
        0.83720242,
        0.72338789,
        1.1006784,
        0.6305293,
        1.29341343,
        0.66171802,
        0.44694119,
        1.15504284,
        0.9962961,
        0.94124575,
        0.59211905,
        0.59794298,
        0.72723468,
        1.0277179,
        0.77173892,
        1.48084607,
        1.11793685,
        1.69110757,
        1.09069987,
        0.88204534,
        1.03605009,
        0.78238945,
        0.67603023,
        0.37089555,
        0.76600735,
        0.73511676,
        1.41916084,
        1.27205298,
        0.92284051,
        0.69979427,
        1.10406604,
        1.35191577,
        0.64020788,
        1.1750396,
        0.55054776,
        0.6170999,
        1.23232973,
        1.62209116,
        1.11793685,
        0.94978794,
        0.91115506,
        1.82874403,
        0.61474962,
        1.02498495,
        0.74222883,
        0.45192687,
        0.72338789,
        0.85715023,
        0.8213718,
        1.71614906,
        1.56089429,
        0.89263066,
        0.89443231,
        1.03047288,
        0.63889148,
        0.79987361,
        0.88378336,
        0.47692251,
        0.75383595,
        0.64965756,
        0.9938061,
        0.7729008,
        0.98163076,
        0.77406797,
        1.14352467,
        0.72918131,
        1.26687639,
        0.87016043,
        0.61592138,
        0.65738468,
        0.8685016,
        0.95196016,
        1.1286906,
        0.73612022,
        1.01958394,
        0.82702364,
        1.10406604,
        0.97220689,
        0.89990358,
        0.8213718,
        1.27729362,
        0.87182882,
        0.75061627,
        1.37107687,
        0.55780883,
        0.93706037,
        0.37102206,
        0.49530439,
        1.04748223,
        1.20486743,
        1.34570526,
        1.65552059,
        1.0563096,
        2.34093267,
        0.30826315,
        0.52304024,
        0.91500063,
        0.51192976,
        0.85242013,
        0.83279414,
        1.04748223,
        0.6170999,
        1.13976129,
        0.6375832,
        0.83720242,
        0.78726719,
        0.52232437,
        0.35027719,
        0.59159805,
        0.83134016,
        2.19359612,
        0.68801289,
        1.11793685,
        0.91889543,
        1.20046671,
        1.35821317,
        0.47264078,
        0.90735768,
        0.96989312,
        0.70415159,
        0.74638708,
        1.1286906,
        0.80773549,
        0.8298938,
        1.24678533,
        1.23709218,
        1.91149701,
        0.62188293,
        1.4264518,
        1.61139043,
        0.70680555,
        1.06536439,
        0.58697047,
        0.78360035,
        0.79987361,
        1.35821317,
        1.09399578,
        2.22083819,
        0.44211861,
        1.43385643,
        0.92483225,
        0.59316522,
        0.53529744,
        1.26176258,
        0.79987361,
        0.80508908,
        2.70298185,
        0.92885482,
        0.42334626,
        1.58051669,
        1.41198065,
        2.80899934,
        0.59316522,
        0.39742608,
        0.43496633,
        0.76714341,
        0.84018057,
        0.70503291,
        1.05930211,
        0.82845498,
        1.31016017,
        0.71773093,
        0.57754936,
        1.39794159,
        0.86685225,
        0.8213718,
        1.54198563,
        0.66465521,
        1.28259966,
        0.74120012,
        0.60174037,
        0.86521229,
        0.76487634,
        1.39794159,
        0.64828182,
        0.65034873,
        1.3645995,
        1.59060977,
        0.89806865,
        0.67603023,
        0.71680099,
        0.49837483,
        0.48385432,
        1.01426758,
        0.67995431,
        1.27205298,
        1.29341343,
        0.74638708,
        0.79098616,
        1.18333468,
        1.48084607,
        0.73915552,
        1.00388016,
        0.82702364,
        0.40042464,
        0.7729008,
        0.33734428,
        0.77524046,
        1.56089429,
        0.64555649,
        1.87706736,
        1.37764747,
        0.99133471,
        0.60729367,
        1.0563096,
        1.11793685,
        0.85874456,
        1.01164046,
        0.92885482,
        1.20046671,
        0.66989123,
        1.57061349,
        0.78481691,
        0.7253036,
        0.93293047,
        1.23709218,
        1.35821317,
        0.75168489,
        1.15116481,
        1.23709218,
        1.05334235,
        1.10748521,
        0.75600564,
        0.73511676,
        1.3645995,
        1.00644701,
        0.76375034,
        0.76040205,
        0.76262932,
        0.84623378,
        1.47269279,
        1.13976129,
        1.46467284,
        1.20931697,
        0.91694181,
        0.85398809,
        1.71614906,
        0.96080087,
        0.43828808,
        0.63563584,
        1.91149701,
        1.06536439,
        1.16291816,
        1.1006784,
        0.42240425,
        1.39794159,
        1.27205298,
        0.55223055,
        0.89263066,
        0.96989312,
        1.02227372,
        1.4264518,
        2.19359612,
        1.59060977,
        0.77879026,
        0.52593367,
        0.63889148,
        1.13976129,
        0.53149183,
        0.55012948,
        0.95414739,
        0.77406797,
        1.05040001,
        1.20486743,
        1.38431355,
        1.44137769,
        1.18333468,
        3.7535231,
        1.07153214,
        1.00644701,
        0.62802199,
        1.37107687,
        0.5380109,
        0.97220689,
        1.4264518,
        0.82559971,
        0.66171802,
        0.96531481,
        1.84443453,
        1.48913641,
        1.51486944,
        1.32757555,
        0.81858911,
        0.90547676,
        0.9563498,
        0.81720837,
        0.77879026,
        0.73915552,
        1.44901869,
        0.4002651,
        0.90925041,
        1.45678263,
        0.42659745,
        1.15504284,
        0.93706037,
        0.71220514,
        0.77641831,
        1.24191034,
        1.53278561,
        0.75168489,
        0.80906855,
        1.09732181,
        4.68137503,
        0.60010403,
        0.63563584,
        0.88905988,
        0.3820034,
        0.83720242,
        1.40490846,
        0.77173892,
        1.00133289,
        0.87689233,
        1.09732181,
        1.14352467,
        0.62927188,
        1.15116481,
        0.87350687,
        1.66713195,
        0.70152743,
        1.13976129,
        0.69382825,
        0.8357252,
        1.10406604,
        0.39931192,
        0.64965756,
        1.20931697,
        3.8951999,
        0.95414739,
        1.23232973,
        0.96759582,
        0.48903843,
        0.75819425,
        0.62309626,
        0.87519467,
        0.55392896,
        3.62626189,
        1.31588883,
        0.46193246,
        0.56719083,
        0.79223762,
        0.65524985,
        0.80377562,
        0.59004331,
        0.57754936,
        1.1109364,
        0.80377562,
        0.79475852,
        1.29892404,
        0.84168166,
        1.21381625,
        0.68801289,
        1.53278561,
        0.52629877,
        1.26687639,
        1.19611393,
        0.45732375,
        1.06536439,
        0.98888171,
        1.20931697,
        1.64414869,
        1.25171825,
        1.53278561,
        0.98163076,
        0.90735768,
        1.12148715,
        0.79987361,
        1.20931697,
        0.66098982,
        0.78481691,
        1.35191577,
        1.03605009,
    ]
)
best_th = np.array(
    [
        0.11,
        0.07,
        0.13,
        0.05,
        0.1,
        0.11,
        0.08,
        0.05,
        0.24,
        0.09,
        0.24,
        0.45,
        0.12,
        0.06,
        0.16,
        0.04,
        0.25,
        0.06,
        0.08,
        0.08,
        0.05,
        0.14,
        0.15,
        0.19,
        0.1,
        0.05,
        0.16,
        0.22,
        0.11,
        0.05,
        0.05,
        0.07,
        0.21,
        0.04,
        0.13,
        0.1,
        0.07,
        0.21,
        0.21,
        0.07,
        0.17,
        0.14,
        0.1,
        0.11,
        0.16,
        0.12,
        0.04,
        0.19,
        0.07,
        0.09,
        0.13,
        0.2,
        0.29,
        0.18,
        0.31,
        0.14,
        0.12,
        0.18,
        0.1,
        0.1,
        0.12,
        0.1,
        0.07,
        0.19,
        0.17,
        0.05,
        0.16,
        0.32,
        0.11,
        0.13,
        0.41,
        0.08,
        0.09,
        0.08,
        0.05,
        0.05,
        0.04,
        0.14,
        0.14,
        0.06,
        0.09,
        0.35,
        0.08,
        0.14,
        0.08,
        0.12,
        0.07,
        0.17,
        0.14,
        0.14,
        0.13,
        0.15,
        0.07,
        0.08,
        0.12,
        0.21,
        0.44,
        0.08,
        0.07,
        0.15,
        0.08,
        0.22,
        0.1,
        0.03,
        0.15,
        0.06,
        0.04,
        0.14,
        0.3,
        0.08,
        0.32,
        0.23,
        0.18,
        0.16,
        0.11,
        0.05,
        0.3,
        0.14,
        0.14,
        0.32,
        0.1,
        0.15,
        0.09,
        0.22,
        0.16,
        0.11,
        0.06,
        0.14,
        0.21,
        0.18,
        0.31,
        0.39,
        0.06,
        0.09,
        0.32,
        0.17,
        0.03,
        0.07,
        0.16,
        0.3,
        0.05,
        0.05,
        0.2,
        0.08,
        0.06,
        0.06,
        0.11,
        0.09,
        0.14,
        0.13,
        0.2,
        0.08,
        0.06,
        0.05,
        0.04,
        0.15,
        0.18,
        0.1,
        0.11,
        0.2,
        0.17,
        0.09,
        0.05,
        0.08,
        0.35,
        0.11,
        0.1,
        0.08,
        0.08,
        0.24,
        0.15,
        0.18,
        0.07,
        0.44,
        0.12,
        0.12,
        0.12,
        0.13,
        0.14,
        0.08,
        0.05,
        0.05,
        0.28,
        0.16,
        0.02,
        0.07,
        0.19,
        0.06,
        0.05,
        0.14,
        0.19,
        0.2,
        0.13,
        0.09,
        0.06,
        0.22,
        0.06,
        0.04,
        0.26,
        0.16,
        0.07,
        0.04,
        0.04,
        0.17,
        0.19,
        0.13,
        0.13,
        0.15,
        0.26,
        0.33,
        0.11,
        0.2,
        0.16,
        0.46,
        0.04,
        0.09,
        0.19,
        0.11,
        0.33,
        0.11,
        0.29,
        0.11,
        0.09,
        0.35,
        0.11,
        0.23,
        0.12,
        0.1,
        0.12,
        0.08,
        0.07,
        0.14,
        0.21,
        0.06,
        0.17,
        0.15,
        0.05,
        0.33,
        0.11,
        0.13,
        0.06,
        0.17,
        0.1,
        0.12,
        0.05,
        0.25,
        0.12,
        0.12,
        0.21,
        0.04,
        0.49,
        0.14,
        0.1,
        0.19,
        0.23,
        0.08,
        0.14,
        0.27,
        0.18,
        0.14,
        0.07,
        0.13,
        0.41,
        0.17,
        0.09,
        0.12,
        0.13,
        0.18,
        0.06,
        0.17,
        0.18,
        0.03,
        0.1,
        0.08,
        0.08,
        0.06,
        0.15,
        0.06,
        0.03,
        0.06,
        0.46,
        0.07,
        0.12,
        0.1,
        0.18,
        0.11,
        0.08,
        0.11,
        0.36,
        0.23,
        0.14,
        0.42,
        0.1,
        0.45,
        0.04,
        0.1,
        0.05,
        0.13,
        0.06,
        0.11,
        0.29,
        0.16,
        0.1,
        0.06,
        0.01,
        0.15,
        0.02,
        0.08,
        0.26,
        0.11,
        0.14,
        0.21,
        0.07,
        0.08,
        0.07,
        0.1,
        0.39,
        0.16,
        0.19,
        0.09,
        0.07,
        0.16,
        0.13,
        0.07,
        0.13,
        0.13,
        0.07,
        0.18,
        0.06,
        0.26,
        0.07,
        0.06,
        0.07,
        0.37,
        0.05,
        0.32,
        0.08,
        0.1,
        0.15,
        0.11,
        0.09,
        0.23,
        0.05,
        0.04,
        0.11,
        0.21,
        0.1,
        0.03,
        0.1,
        0.3,
        0.13,
        0.2,
        0.03,
        0.06,
        0.07,
        0.19,
        0.16,
        0.25,
        0.18,
        0.24,
        0.1,
        0.11,
        0.14,
        0.1,
        0.11,
        0.05,
        0.06,
        0.07,
        0.2,
        0.26,
        0.11,
        0.14,
        0.23,
        0.05,
        0.13,
        0.22,
        0.1,
        0.14,
        0.05,
        0.05,
        0.12,
        0.04,
        0.1,
        0.08,
        0.07,
        0.22,
        0.2,
        0.12,
        0.09,
        0.1,
        0.16,
        0.2,
        0.07,
        0.17,
        0.17,
        0.09,
        0.13,
    ]
)
target_columns = [
    "acafly",
    "acowoo",
    "aldfly",
    "ameavo",
    "amecro",
    "amegfi",
    "amekes",
    "amepip",
    "amered",
    "amerob",
    "amewig",
    "amtspa",
    "andsol1",
    "annhum",
    "astfly",
    "azaspi1",
    "babwar",
    "baleag",
    "balori",
    "banana",
    "banswa",
    "banwre1",
    "barant1",
    "barswa",
    "batpig1",
    "bawswa1",
    "bawwar",
    "baywre1",
    "bbwduc",
    "bcnher",
    "belkin1",
    "belvir",
    "bewwre",
    "bkbmag1",
    "bkbplo",
    "bkbwar",
    "bkcchi",
    "bkhgro",
    "bkmtou1",
    "bknsti",
    "blbgra1",
    "blbthr1",
    "blcjay1",
    "blctan1",
    "blhpar1",
    "blkpho",
    "blsspa1",
    "blugrb1",
    "blujay",
    "bncfly",
    "bnhcow",
    "bobfly1",
    "bongul",
    "botgra",
    "brbmot1",
    "brbsol1",
    "brcvir1",
    "brebla",
    "brncre",
    "brnjay",
    "brnthr",
    "brratt1",
    "brwhaw",
    "brwpar1",
    "btbwar",
    "btnwar",
    "btywar",
    "bucmot2",
    "buggna",
    "bugtan",
    "buhvir",
    "bulori",
    "burwar1",
    "bushti",
    "butsal1",
    "buwtea",
    "cacgoo1",
    "cacwre",
    "calqua",
    "caltow",
    "cangoo",
    "canwar",
    "carchi",
    "carwre",
    "casfin",
    "caskin",
    "caster1",
    "casvir",
    "categr",
    "ccbfin",
    "cedwax",
    "chbant1",
    "chbchi",
    "chbwre1",
    "chcant2",
    "chispa",
    "chswar",
    "cinfly2",
    "clanut",
    "clcrob",
    "cliswa",
    "cobtan1",
    "cocwoo1",
    "cogdov",
    "colcha1",
    "coltro1",
    "comgol",
    "comgra",
    "comloo",
    "commer",
    "compau",
    "compot1",
    "comrav",
    "comyel",
    "coohaw",
    "cotfly1",
    "cowscj1",
    "cregua1",
    "creoro1",
    "crfpar",
    "cubthr",
    "daejun",
    "dowwoo",
    "ducfly",
    "dusfly",
    "easblu",
    "easkin",
    "easmea",
    "easpho",
    "eastow",
    "eawpew",
    "eletro",
    "eucdov",
    "eursta",
    "fepowl",
    "fiespa",
    "flrtan1",
    "foxspa",
    "gadwal",
    "gamqua",
    "gartro1",
    "gbbgul",
    "gbwwre1",
    "gcrwar",
    "gilwoo",
    "gnttow",
    "gnwtea",
    "gocfly1",
    "gockin",
    "gocspa",
    "goftyr1",
    "gohque1",
    "goowoo1",
    "grasal1",
    "grbani",
    "grbher3",
    "grcfly",
    "greegr",
    "grekis",
    "grepew",
    "grethr1",
    "gretin1",
    "greyel",
    "grhcha1",
    "grhowl",
    "grnher",
    "grnjay",
    "grtgra",
    "grycat",
    "gryhaw2",
    "gwfgoo",
    "haiwoo",
    "heptan",
    "hergul",
    "herthr",
    "herwar",
    "higmot1",
    "hofwoo1",
    "houfin",
    "houspa",
    "houwre",
    "hutvir",
    "incdov",
    "indbun",
    "kebtou1",
    "killde",
    "labwoo",
    "larspa",
    "laufal1",
    "laugul",
    "lazbun",
    "leafly",
    "leasan",
    "lesgol",
    "lesgre1",
    "lesvio1",
    "linspa",
    "linwoo1",
    "littin1",
    "lobdow",
    "lobgna5",
    "logshr",
    "lotduc",
    "lotman1",
    "lucwar",
    "macwar",
    "magwar",
    "mallar3",
    "marwre",
    "mastro1",
    "meapar",
    "melbla1",
    "monoro1",
    "mouchi",
    "moudov",
    "mouela1",
    "mouqua",
    "mouwar",
    "mutswa",
    "naswar",
    "norcar",
    "norfli",
    "normoc",
    "norpar",
    "norsho",
    "norwat",
    "nrwswa",
    "nutwoo",
    "oaktit",
    "obnthr1",
    "ocbfly1",
    "oliwoo1",
    "olsfly",
    "orbeup1",
    "orbspa1",
    "orcpar",
    "orcwar",
    "orfpar",
    "osprey",
    "ovenbi1",
    "pabspi1",
    "paltan1",
    "palwar",
    "pasfly",
    "pavpig2",
    "phivir",
    "pibgre",
    "pilwoo",
    "pinsis",
    "pirfly1",
    "plawre1",
    "plaxen1",
    "plsvir",
    "plupig2",
    "prowar",
    "purfin",
    "purgal2",
    "putfru1",
    "pygnut",
    "rawwre1",
    "rcatan1",
    "rebnut",
    "rebsap",
    "rebwoo",
    "redcro",
    "reevir1",
    "rehbar1",
    "relpar",
    "reshaw",
    "rethaw",
    "rewbla",
    "ribgul",
    "rinkin1",
    "roahaw",
    "robgro",
    "rocpig",
    "rotbec",
    "royter1",
    "rthhum",
    "rtlhum",
    "ruboro1",
    "rubpep1",
    "rubrob",
    "rubwre1",
    "ruckin",
    "rucspa1",
    "rucwar",
    "rucwar1",
    "rudpig",
    "rudtur",
    "rufhum",
    "rugdov",
    "rumfly1",
    "runwre1",
    "rutjac1",
    "saffin",
    "sancra",
    "sander",
    "savspa",
    "saypho",
    "scamac1",
    "scatan",
    "scbwre1",
    "scptyr1",
    "scrtan1",
    "semplo",
    "shicow",
    "sibtan2",
    "sinwre1",
    "sltred",
    "smbani",
    "snogoo",
    "sobtyr1",
    "socfly1",
    "solsan",
    "sonspa",
    "soulap1",
    "sposan",
    "spotow",
    "spvear1",
    "squcuc1",
    "stbori",
    "stejay",
    "sthant1",
    "sthwoo1",
    "strcuc1",
    "strfly1",
    "strsal1",
    "stvhum2",
    "subfly",
    "sumtan",
    "swaspa",
    "swathr",
    "tenwar",
    "thbeup1",
    "thbkin",
    "thswar1",
    "towsol",
    "treswa",
    "trogna1",
    "trokin",
    "tromoc",
    "tropar",
    "tropew1",
    "tuftit",
    "tunswa",
    "veery",
    "verdin",
    "vigswa",
    "warvir",
    "wbwwre1",
    "webwoo1",
    "wegspa1",
    "wesant1",
    "wesblu",
    "weskin",
    "wesmea",
    "westan",
    "wewpew",
    "whbman1",
    "whbnut",
    "whcpar",
    "whcsee1",
    "whcspa",
    "whevir",
    "whfpar1",
    "whimbr",
    "whiwre1",
    "whtdov",
    "whtspa",
    "whwbec1",
    "whwdov",
    "wilfly",
    "willet1",
    "wilsni1",
    "wiltur",
    "wlswar",
    "wooduc",
    "woothr",
    "wrenti",
    "y00475",
    "yebcha",
    "yebela1",
    "yebfly",
    "yebori1",
    "yebsap",
    "yebsee1",
    "yefgra1",
    "yegvir",
    "yehbla",
    "yehcar1",
    "yelgro",
    "yelwar",
    "yeofly1",
    "yerwar",
    "yeteup1",
    "yetvir",
]

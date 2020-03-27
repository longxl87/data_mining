# -*- coding: utf-8 -*-

import numpy as np
import feature.feature
import pandas as pd
data = pd.read_csv('D:/Github/data_mining/test/tmp.csv')


print(feature.feature.woe_code(data.tencent_af_risk_score,data.f_fpd10))
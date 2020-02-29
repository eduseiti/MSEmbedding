import os
import pandas as pd
import json
import time
import pickle

import datetime
import math

import numpy as np
import re

import torch

from bootstrap.lib.logger import Logger

from .spectra import Scan
from .spectra import SpectraFound
from .spectra import MGF


class PXD000561:

#
# This initial set of files depends on the identifications provided by the original work, in the form of .msf files converted to .csv
#

    ADULT_ADRENALGLAND_GEL_ELITE_FILES = {
        'b01' : 'Adult_Adrenalgland_Gel_Elite_49_f01.mgf',
        'b02' : 'Adult_Adrenalgland_Gel_Elite_49_f02.mgf',
        'b03' : 'Adult_Adrenalgland_Gel_Elite_49_f03.mgf',
        'b04' : 'Adult_Adrenalgland_Gel_Elite_49_f04.mgf',
        'b05' : 'Adult_Adrenalgland_Gel_Elite_49_f05.mgf',
        'b06' : 'Adult_Adrenalgland_Gel_Elite_49_f06.mgf',
        'b07' : 'Adult_Adrenalgland_Gel_Elite_49_f07.mgf',
        'b08' : 'Adult_Adrenalgland_Gel_Elite_49_f08.mgf',
        'b09' : 'Adult_Adrenalgland_Gel_Elite_49_f09.mgf',
        'b10' : 'Adult_Adrenalgland_Gel_Elite_49_f10.mgf',
        'b11' : 'Adult_Adrenalgland_Gel_Elite_49_f11.mgf',
        'b12' : 'Adult_Adrenalgland_Gel_Elite_49_f12.mgf',
        'b13' : 'Adult_Adrenalgland_Gel_Elite_49_f13.mgf',
        'b14' : 'Adult_Adrenalgland_Gel_Elite_49_f14.mgf',
        'b15' : 'Adult_Adrenalgland_Gel_Elite_49_f15.mgf',
        'b16' : 'Adult_Adrenalgland_Gel_Elite_49_f16.mgf',
        'b17' : 'Adult_Adrenalgland_Gel_Elite_49_f17.mgf',
        'b18' : 'Adult_Adrenalgland_Gel_Elite_49_f18.mgf',
        'b19' : 'Adult_Adrenalgland_Gel_Elite_49_f19.mgf',
        'b20' : 'Adult_Adrenalgland_Gel_Elite_49_f20.mgf',
        'b21' : 'Adult_Adrenalgland_Gel_Elite_49_f21.mgf',
        'b22' : 'Adult_Adrenalgland_Gel_Elite_49_f22.mgf',
        'b23' : 'Adult_Adrenalgland_Gel_Elite_49_f23.mgf',
        'b24' : 'Adult_Adrenalgland_Gel_Elite_49_f24.mgf'
    }


    ADULT_ADRENALGLAND_GEL_VELOS_FILES = {
        'D01' : 'Adult_Adrenalgland_Gel_Velos_2_f01.mgf',
        'D02' : 'Adult_Adrenalgland_Gel_Velos_2_f02.mgf',
        'D03' : 'Adult_Adrenalgland_Gel_Velos_2_f03.mgf',
        'D04' : 'Adult_Adrenalgland_Gel_Velos_2_f04.mgf',
        'D05' : 'Adult_Adrenalgland_Gel_Velos_2_f05.mgf',
        'D06' : 'Adult_Adrenalgland_Gel_Velos_2_f06.mgf',
        'D07' : 'Adult_Adrenalgland_Gel_Velos_2_f07.mgf',
        'D08' : 'Adult_Adrenalgland_Gel_Velos_2_f08.mgf',
        'D09' : 'Adult_Adrenalgland_Gel_Velos_2_f09.mgf',
        'D10' : 'Adult_Adrenalgland_Gel_Velos_2_f10.mgf',
        'D11' : 'Adult_Adrenalgland_Gel_Velos_2_f11.mgf',
        'D12' : 'Adult_Adrenalgland_Gel_Velos_2_f12.mgf',
        'E01' : 'Adult_Adrenalgland_Gel_Velos_2_f13.mgf',
        'E01Re' : 'Adult_Adrenalgland_Gel_Velos_2_f14.mgf',
        'E02' : 'Adult_Adrenalgland_Gel_Velos_2_f15.mgf',
        'E03' : 'Adult_Adrenalgland_Gel_Velos_2_f16.mgf',
        'E04' : 'Adult_Adrenalgland_Gel_Velos_2_f17.mgf',
        'E05' : 'Adult_Adrenalgland_Gel_Velos_2_f18.mgf',
        'E06' : 'Adult_Adrenalgland_Gel_Velos_2_f19.mgf',
        'E06-110108181623' : 'Adult_Adrenalgland_Gel_Velos_2_f20.mgf',
        'E07' : 'Adult_Adrenalgland_Gel_Velos_2_f21.mgf',
        'E08' : 'Adult_Adrenalgland_Gel_Velos_2_f22.mgf',
        'E09' : 'Adult_Adrenalgland_Gel_Velos_2_f23.mgf',
        'E10' : 'Adult_Adrenalgland_Gel_Velos_2_f24.mgf',
        'E11' : 'Adult_Adrenalgland_Gel_Velos_2_f25.mgf',
        'E12' : 'Adult_Adrenalgland_Gel_Velos_2_f26.mgf'
    }


    ADULT_ADRENALGLAND_BRP_VELOS_FILES = {
        'A01' : 'Adult_Adrenalgland_bRP_Velos_1_f01.mgf',
        'A02' : 'Adult_Adrenalgland_bRP_Velos_1_f02.mgf',
        'A03' : 'Adult_Adrenalgland_bRP_Velos_1_f03.mgf',
        'A04' : 'Adult_Adrenalgland_bRP_Velos_1_f04.mgf',
        'A05' : 'Adult_Adrenalgland_bRP_Velos_1_f05.mgf',
        'A06' : 'Adult_Adrenalgland_bRP_Velos_1_f06.mgf',
        'A07' : 'Adult_Adrenalgland_bRP_Velos_1_f07.mgf',
        'A08' : 'Adult_Adrenalgland_bRP_Velos_1_f08.mgf',
        'A09' : 'Adult_Adrenalgland_bRP_Velos_1_f09.mgf',
        'A10' : 'Adult_Adrenalgland_bRP_Velos_1_f10.mgf',
        'A11' : 'Adult_Adrenalgland_bRP_Velos_1_f11.mgf',
        'A12' : 'Adult_Adrenalgland_bRP_Velos_1_f12.mgf',
        'B01' : 'Adult_Adrenalgland_bRP_Velos_1_f13.mgf',
        'B02' : 'Adult_Adrenalgland_bRP_Velos_1_f14.mgf',
        'B03' : 'Adult_Adrenalgland_bRP_Velos_1_f15.mgf',
        'B04' : 'Adult_Adrenalgland_bRP_Velos_1_f16.mgf',
        'B05' : 'Adult_Adrenalgland_bRP_Velos_1_f17.mgf',
        'B06' : 'Adult_Adrenalgland_bRP_Velos_1_f18.mgf',
        'B07' : 'Adult_Adrenalgland_bRP_Velos_1_f19.mgf',
        'B08' : 'Adult_Adrenalgland_bRP_Velos_1_f20.mgf',
        'B09' : 'Adult_Adrenalgland_bRP_Velos_1_f21.mgf',
        'B10' : 'Adult_Adrenalgland_bRP_Velos_1_f22.mgf',
        'B11' : 'Adult_Adrenalgland_bRP_Velos_1_f23.mgf',
        'B12' : 'Adult_Adrenalgland_bRP_Velos_1_f24.mgf',
        'C01' : 'Adult_Adrenalgland_bRP_Velos_1_f25.mgf',
        'C02' : 'Adult_Adrenalgland_bRP_Velos_1_f26.mgf',
        'C03' : 'Adult_Adrenalgland_bRP_Velos_1_f27.mgf',
        'C04' : 'Adult_Adrenalgland_bRP_Velos_1_f28.mgf',
        'C05' : 'Adult_Adrenalgland_bRP_Velos_1_f29.mgf',
        'C06' : 'Adult_Adrenalgland_bRP_Velos_1_f30.mgf',
        'C07' : 'Adult_Adrenalgland_bRP_Velos_1_f31.mgf',
        'C08' : 'Adult_Adrenalgland_bRP_Velos_1_f32.mgf',
        'C09' : 'Adult_Adrenalgland_bRP_Velos_1_f33.mgf',
        'C10' : 'Adult_Adrenalgland_bRP_Velos_1_f34.mgf',
        'C11' : 'Adult_Adrenalgland_bRP_Velos_1_f35.mgf',
        'C12' : 'Adult_Adrenalgland_bRP_Velos_1_f36.mgf'
    }


    ADULT_HEART_BRP_ELITE_FILES = {
        'f01' : 'Adult_Heart_bRP_Elite_81_f01.mgf',
        'f02' : 'Adult_Heart_bRP_Elite_81_f02.mgf',
        'f03' : 'Adult_Heart_bRP_Elite_81_f03.mgf',
        'f04' : 'Adult_Heart_bRP_Elite_81_f04.mgf',
        'f05' : 'Adult_Heart_bRP_Elite_81_f05.mgf',
        'f06' : 'Adult_Heart_bRP_Elite_81_f06.mgf',
        'f07' : 'Adult_Heart_bRP_Elite_81_f07.mgf',
        'f08' : 'Adult_Heart_bRP_Elite_81_f08.mgf',
        'f09' : 'Adult_Heart_bRP_Elite_81_f09.mgf',
        'f10' : 'Adult_Heart_bRP_Elite_81_f10.mgf',
        'f11' : 'Adult_Heart_bRP_Elite_81_f11.mgf',
        'f12' : 'Adult_Heart_bRP_Elite_81_f12.mgf',
        'f13' : 'Adult_Heart_bRP_Elite_81_f13.mgf',
        'f14' : 'Adult_Heart_bRP_Elite_81_f14.mgf',
        'f15' : 'Adult_Heart_bRP_Elite_81_f15.mgf',
        'f16' : 'Adult_Heart_bRP_Elite_81_f16.mgf',
        'f17' : 'Adult_Heart_bRP_Elite_81_f17.mgf',
        'f18' : 'Adult_Heart_bRP_Elite_81_f18.mgf',
        'f19' : 'Adult_Heart_bRP_Elite_81_f19.mgf',
        'f20' : 'Adult_Heart_bRP_Elite_81_f20.mgf',
        'f21' : 'Adult_Heart_bRP_Elite_81_f21.mgf',
        'f22' : 'Adult_Heart_bRP_Elite_81_f22.mgf',
        'f23' : 'Adult_Heart_bRP_Elite_81_f23.mgf',
        'f24' : 'Adult_Heart_bRP_Elite_81_f24.mgf'
    }


    ADULT_HEART_BRP_VELOS_FILES = {
        'A1' : 'Adult_Heart_bRP_Velos_6_f01.mgf',
        'A2' : 'Adult_Heart_bRP_Velos_6_f02.mgf',
        'A3' : 'Adult_Heart_bRP_Velos_6_f03.mgf',
        'A4' : 'Adult_Heart_bRP_Velos_6_f04.mgf',
        'A5' : 'Adult_Heart_bRP_Velos_6_f05.mgf',
        'A6' : 'Adult_Heart_bRP_Velos_6_f06.mgf',
        'A7' : 'Adult_Heart_bRP_Velos_6_f07.mgf',
        'A8' : 'Adult_Heart_bRP_Velos_6_f08.mgf',
        'A9' : 'Adult_Heart_bRP_Velos_6_f09.mgf',
        'A10' : 'Adult_Heart_bRP_Velos_6_f10.mgf',
        'A11' : 'Adult_Heart_bRP_Velos_6_f11.mgf',
        'A11-110318011425' : 'Adult_Heart_bRP_Velos_6_f12.mgf',
        'A12' : 'Adult_Heart_bRP_Velos_6_f13.mgf',
        'B1' : 'Adult_Heart_bRP_Velos_6_f14.mgf',
        'B2' : 'Adult_Heart_bRP_Velos_6_f15.mgf',
        'B3' : 'Adult_Heart_bRP_Velos_6_f16.mgf',
        'B1-B3-pooled' : 'Adult_Heart_bRP_Velos_6_f17.mgf',
        'B4' : 'Adult_Heart_bRP_Velos_6_f18.mgf',
        'B5' : 'Adult_Heart_bRP_Velos_6_f19.mgf',
        'B6' : 'Adult_Heart_bRP_Velos_6_f20.mgf',
        'B7' : 'Adult_Heart_bRP_Velos_6_f21.mgf',
        'B8' : 'Adult_Heart_bRP_Velos_6_f22.mgf',
        'B9' : 'Adult_Heart_bRP_Velos_6_f23.mgf',
        'B10' : 'Adult_Heart_bRP_Velos_6_f24.mgf',
        'B10' : 'Adult_Heart_bRP_Velos_6_f25.mgf',
        'B10' : 'Adult_Heart_bRP_Velos_6_f26.mgf'
    }


    ADULT_PLATELETS_GEL_ELITE_FILES = {
        'b01' : 'Adult_Platelets_Gel_Elite_48_f01.mgf',
        'b02' : 'Adult_Platelets_Gel_Elite_48_f02.mgf',
        'b03' : 'Adult_Platelets_Gel_Elite_48_f03.mgf',
        'b04' : 'Adult_Platelets_Gel_Elite_48_f04.mgf',
        'b05' : 'Adult_Platelets_Gel_Elite_48_f05.mgf',
        'b06' : 'Adult_Platelets_Gel_Elite_48_f06.mgf',
        'b07' : 'Adult_Platelets_Gel_Elite_48_f07.mgf',
        'b08' : 'Adult_Platelets_Gel_Elite_48_f08.mgf',
        'b09' : 'Adult_Platelets_Gel_Elite_48_f09.mgf',
        'b10' : 'Adult_Platelets_Gel_Elite_48_f10.mgf',
        'b11' : 'Adult_Platelets_Gel_Elite_48_f11.mgf',
        'b12' : 'Adult_Platelets_Gel_Elite_48_f12.mgf',
        'b13' : 'Adult_Platelets_Gel_Elite_48_f13.mgf',
        'b14' : 'Adult_Platelets_Gel_Elite_48_f14.mgf',
        'b15' : 'Adult_Platelets_Gel_Elite_48_f15.mgf',
        'b16' : 'Adult_Platelets_Gel_Elite_48_f16.mgf',
        'b17' : 'Adult_Platelets_Gel_Elite_48_f17.mgf',
        'b18' : 'Adult_Platelets_Gel_Elite_48_f18.mgf',
        'b19' : 'Adult_Platelets_Gel_Elite_48_f19.mgf',
        'b20' : 'Adult_Platelets_Gel_Elite_48_f20.mgf',
        'b21' : 'Adult_Platelets_Gel_Elite_48_f21.mgf',
        'b22' : 'Adult_Platelets_Gel_Elite_48_f22.mgf',
        'b23' : 'Adult_Platelets_Gel_Elite_48_f23.mgf',
        'b24' : 'Adult_Platelets_Gel_Elite_48_f24.mgf',
    }


    ADULT_URINARYBLADDER_GEL_ELITE_FILES = {
        'b01' : 'Adult_Urinarybladder_Gel_Elite_70_f01.mgf',
        'b02' : 'Adult_Urinarybladder_Gel_Elite_70_f02.mgf',
        'b03' : 'Adult_Urinarybladder_Gel_Elite_70_f03.mgf',
        'b04' : 'Adult_Urinarybladder_Gel_Elite_70_f04.mgf',
        'b05' : 'Adult_Urinarybladder_Gel_Elite_70_f05.mgf',
        'b06' : 'Adult_Urinarybladder_Gel_Elite_70_f06.mgf',
        'b07' : 'Adult_Urinarybladder_Gel_Elite_70_f07.mgf',
        'b08' : 'Adult_Urinarybladder_Gel_Elite_70_f08.mgf',
        'b09' : 'Adult_Urinarybladder_Gel_Elite_70_f09.mgf',
        'b09r' : 'Adult_Urinarybladder_Gel_Elite_70_f10.mgf',
        'b10' : 'Adult_Urinarybladder_Gel_Elite_70_f11.mgf',
        'b11' : 'Adult_Urinarybladder_Gel_Elite_70_f12.mgf',
        'b12' : 'Adult_Urinarybladder_Gel_Elite_70_f13.mgf',
        'b13' : 'Adult_Urinarybladder_Gel_Elite_70_f14.mgf',
        'b14' : 'Adult_Urinarybladder_Gel_Elite_70_f15.mgf',
        'b15' : 'Adult_Urinarybladder_Gel_Elite_70_f16.mgf',
        'b16' : 'Adult_Urinarybladder_Gel_Elite_70_f17.mgf',
        'b17' : 'Adult_Urinarybladder_Gel_Elite_70_f18.mgf',
        'b18' : 'Adult_Urinarybladder_Gel_Elite_70_f19.mgf',
        'b19' : 'Adult_Urinarybladder_Gel_Elite_70_f20.mgf',
        'b20' : 'Adult_Urinarybladder_Gel_Elite_70_f21.mgf',
        'b21' : 'Adult_Urinarybladder_Gel_Elite_70_f22.mgf',
        'b22' : 'Adult_Urinarybladder_Gel_Elite_70_f23.mgf',
        'b23' : 'Adult_Urinarybladder_Gel_Elite_70_f24.mgf',
        'b24' : 'Adult_Urinarybladder_Gel_Elite_70_f25.mgf',
    }


    FETAL_BRAIN_GEL_VELOS_FILES = {
		"E01" : "Fetal_Brain_Gel_Velos_16_f01.mgf",
		"E01rep" : "Fetal_Brain_Gel_Velos_16_f02.mgf",
		"E02" : "Fetal_Brain_Gel_Velos_16_f03.mgf",
		"E03" : "Fetal_Brain_Gel_Velos_16_f04.mgf",
		"E03rep" : "Fetal_Brain_Gel_Velos_16_f05.mgf",
		"E04" : "Fetal_Brain_Gel_Velos_16_f06.mgf",
		"E05" : "Fetal_Brain_Gel_Velos_16_f07.mgf",
		"E06" : "Fetal_Brain_Gel_Velos_16_f08.mgf",
		"E07" : "Fetal_Brain_Gel_Velos_16_f09.mgf",
		"E08" : "Fetal_Brain_Gel_Velos_16_f10.mgf",
		"E09" : "Fetal_Brain_Gel_Velos_16_f11.mgf",
		"E10" : "Fetal_Brain_Gel_Velos_16_f12.mgf",
		"E11" : "Fetal_Brain_Gel_Velos_16_f13.mgf",
		"E12" : "Fetal_Brain_Gel_Velos_16_f14.mgf",
		"F01" : "Fetal_Brain_Gel_Velos_16_f15.mgf",
		"F02" : "Fetal_Brain_Gel_Velos_16_f16.mgf",
		"F03" : "Fetal_Brain_Gel_Velos_16_f17.mgf",
		"F04" : "Fetal_Brain_Gel_Velos_16_f18.mgf",
		"F05" : "Fetal_Brain_Gel_Velos_16_f19.mgf",
		"F06" : "Fetal_Brain_Gel_Velos_16_f20.mgf",
		"F07" : "Fetal_Brain_Gel_Velos_16_f21.mgf",
		"F08" : "Fetal_Brain_Gel_Velos_16_f22.mgf",
		"F09" : "Fetal_Brain_Gel_Velos_16_f23.mgf",
		"F10" : "Fetal_Brain_Gel_Velos_16_f24.mgf",
		"F11" : "Fetal_Brain_Gel_Velos_16_f25.mgf",
		"F12" : "Fetal_Brain_Gel_Velos_16_f26.mgf",
		"G01" : "Fetal_Brain_Gel_Velos_16_f27.mgf",
		"G02" : "Fetal_Brain_Gel_Velos_16_f28.mgf",
		"G03" : "Fetal_Brain_Gel_Velos_16_f29.mgf"
    }


    FETAL_LIVER_BRP_ELITE_FILES = {
        '01' : 'Fetal_Liver_bRP_Elite_22_f01.mgf',
        '02' : 'Fetal_Liver_bRP_Elite_22_f02.mgf',
        '03' : 'Fetal_Liver_bRP_Elite_22_f03.mgf',
        '04' : 'Fetal_Liver_bRP_Elite_22_f04.mgf',
        '05' : 'Fetal_Liver_bRP_Elite_22_f05.mgf',
        '06' : 'Fetal_Liver_bRP_Elite_22_f06.mgf',
        '07' : 'Fetal_Liver_bRP_Elite_22_f07.mgf',
        '08' : 'Fetal_Liver_bRP_Elite_22_f08.mgf',
        '08-NCE27' : 'Fetal_Liver_bRP_Elite_22_f09.mgf',
        '08-NCE27-1' : 'Fetal_Liver_bRP_Elite_22_f10.mgf',
        '08-NCE27-32' : 'Fetal_Liver_bRP_Elite_22_f11.mgf',
        '08-NCE27-32-1' : 'Fetal_Liver_bRP_Elite_22_f12.mgf',
        '08-1' : 'Fetal_Liver_bRP_Elite_22_f13.mgf',
        '09' : 'Fetal_Liver_bRP_Elite_22_f14.mgf',
        '10' : 'Fetal_Liver_bRP_Elite_22_f15.mgf',
        '11' : 'Fetal_Liver_bRP_Elite_22_f16.mgf',
        '12' : 'Fetal_Liver_bRP_Elite_22_f17.mgf',
        '13-120502235957' : 'Fetal_Liver_bRP_Elite_22_f18.mgf',
        '13' : 'Fetal_Liver_bRP_Elite_22_f19.mgf',
        '14-120503014940' : 'Fetal_Liver_bRP_Elite_22_f20.mgf',
        '14' : 'Fetal_Liver_bRP_Elite_22_f21.mgf',
        '15-120503033941' : 'Fetal_Liver_bRP_Elite_22_f22.mgf',
        '15' : 'Fetal_Liver_bRP_Elite_22_f23.mgf',
        '16-120503052933' : 'Fetal_Liver_bRP_Elite_22_f24.mgf',
        '16' : 'Fetal_Liver_bRP_Elite_22_f25.mgf',
        '17-120503071920' : 'Fetal_Liver_bRP_Elite_22_f26.mgf',
        '17' : 'Fetal_Liver_bRP_Elite_22_f27.mgf',
        '18' : 'Fetal_Liver_bRP_Elite_22_f28.mgf',
        '19' : 'Fetal_Liver_bRP_Elite_22_f29.mgf',
        '20' : 'Fetal_Liver_bRP_Elite_22_f30.mgf',
        '21' : 'Fetal_Liver_bRP_Elite_22_f31.mgf',
        '22' : 'Fetal_Liver_bRP_Elite_22_f32.mgf',
        '23401' : 'Fetal_Liver_bRP_Elite_22_f33.mgf',
        '23' : 'Fetal_Liver_bRP_Elite_22_f34.mgf',
        '24' : 'Fetal_Liver_bRP_Elite_22_f35.mgf',
    }


    FETAL_OVARY_BRP_VELOS_FILES = {
        'E01' : 'Fetal_Ovary_bRP_Velos_41_f01.mgf',
        'E02' : 'Fetal_Ovary_bRP_Velos_41_f02.mgf',
        'E03' : 'Fetal_Ovary_bRP_Velos_41_f03.mgf',
        'E04' : 'Fetal_Ovary_bRP_Velos_41_f04.mgf',
        'E05' : 'Fetal_Ovary_bRP_Velos_41_f05.mgf',
        'E06' : 'Fetal_Ovary_bRP_Velos_41_f06.mgf',
        'E06-E1-E6' : 'Fetal_Ovary_bRP_Velos_41_f07.mgf',
        'E07' : 'Fetal_Ovary_bRP_Velos_41_f08.mgf',
        'E08' : 'Fetal_Ovary_bRP_Velos_41_f09.mgf',
        'E09' : 'Fetal_Ovary_bRP_Velos_41_f26.mgf',
        'E11' : 'Fetal_Ovary_bRP_Velos_41_f10.mgf',
        'E12' : 'Fetal_Ovary_bRP_Velos_41_f11.mgf',
        'E12-E7-E12' : 'Fetal_Ovary_bRP_Velos_41_f12.mgf',
        'F01' : 'Fetal_Ovary_bRP_Velos_41_f13.mgf',
        'F02' : 'Fetal_Ovary_bRP_Velos_41_f14.mgf',
        'F03' : 'Fetal_Ovary_bRP_Velos_41_f15.mgf',
        'F04' : 'Fetal_Ovary_bRP_Velos_41_f16.mgf',
        'F05' : 'Fetal_Ovary_bRP_Velos_41_f17.mgf',
        'F06' : 'Fetal_Ovary_bRP_Velos_41_f18.mgf',
        'F06-F1-F6' : 'Fetal_Ovary_bRP_Velos_41_f19.mgf',
        'F07' : 'Fetal_Ovary_bRP_Velos_41_f20.mgf',
        'F08' : 'Fetal_Ovary_bRP_Velos_41_f21.mgf',
        'F09' : 'Fetal_Ovary_bRP_Velos_41_f22.mgf',
        'F10' : 'Fetal_Ovary_bRP_Velos_41_f23.mgf',
        'F11' : 'Fetal_Ovary_bRP_Velos_41_f24.mgf',
        'F12' : 'Fetal_Ovary_bRP_Velos_41_f25.mgf',
    }


    FETAL_OVARY_BRP_ELITE_FILES = {
        '01' : 'Fetal_Ovary_bRP_Elite_25_f01.mgf',
        '02' : 'Fetal_Ovary_bRP_Elite_25_f02.mgf',
        '03' : 'Fetal_Ovary_bRP_Elite_25_f03.mgf',
        '04' : 'Fetal_Ovary_bRP_Elite_25_f04.mgf',
        '05' : 'Fetal_Ovary_bRP_Elite_25_f05.mgf',
        '06' : 'Fetal_Ovary_bRP_Elite_25_f06.mgf',
        '07' : 'Fetal_Ovary_bRP_Elite_25_f07.mgf',
        '08' : 'Fetal_Ovary_bRP_Elite_25_f08.mgf',
        '09' : 'Fetal_Ovary_bRP_Elite_25_f09.mgf',
        '10' : 'Fetal_Ovary_bRP_Elite_25_f10.mgf',
        '11' : 'Fetal_Ovary_bRP_Elite_25_f11.mgf',
        '12' : 'Fetal_Ovary_bRP_Elite_25_f12.mgf',
        '13' : 'Fetal_Ovary_bRP_Elite_25_f13.mgf',
        '14' : 'Fetal_Ovary_bRP_Elite_25_f14.mgf',
        '15' : 'Fetal_Ovary_bRP_Elite_25_f15.mgf',
        '16' : 'Fetal_Ovary_bRP_Elite_25_f16.mgf',
        '17' : 'Fetal_Ovary_bRP_Elite_25_f17.mgf',
        '18' : 'Fetal_Ovary_bRP_Elite_25_f18.mgf',
        '19' : 'Fetal_Ovary_bRP_Elite_25_f19.mgf',
        '20' : 'Fetal_Ovary_bRP_Elite_25_f20.mgf',
        '21' : 'Fetal_Ovary_bRP_Elite_25_f21.mgf',
        '22' : 'Fetal_Ovary_bRP_Elite_25_f22.mgf',
        '23' : 'Fetal_Ovary_bRP_Elite_25_f23.mgf',
        '24' : 'Fetal_Ovary_bRP_Elite_25_f24.mgf',
    }



    #
    # This dictionary maps the identifications file into the corresponding experimental .mgf spectra
    #

    MATCHES_TO_FILES_LIST = {
        "adult_adrenalgland_gel_elite.csv" : ADULT_ADRENALGLAND_GEL_ELITE_FILES,
        "adult_adrenalgland_gel_velos.csv" : ADULT_ADRENALGLAND_GEL_VELOS_FILES,
        "adult_adrenalgland_bRP_velos.csv" : ADULT_ADRENALGLAND_BRP_VELOS_FILES,
        "adult_heart_brp_elite.csv" : ADULT_HEART_BRP_ELITE_FILES,
        "adult_heart_brp_velos.csv" : ADULT_HEART_BRP_VELOS_FILES,
        "adult_platelets_gel_elite.csv" : ADULT_PLATELETS_GEL_ELITE_FILES,
        "adult_urinarybladder_gel_elite.csv" : ADULT_URINARYBLADDER_GEL_ELITE_FILES,
        "fetal_brain_gel_velos.csv" : FETAL_BRAIN_GEL_VELOS_FILES,
        "fetal_ovary_brp_velos.csv" : FETAL_OVARY_BRP_VELOS_FILES,
        "fetal_ovary_brp_elite.csv" : FETAL_OVARY_BRP_ELITE_FILES
    }




#
# This second set of files are used with identification files created using crux tool
#

    ADULT_ADRENALGLAND_GEL_ELITE_CRUX_FILES = [
        'Adult_Adrenalgland_Gel_Elite_49_f01.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f02.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f03.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f04.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f05.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f06.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f07.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f08.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f09.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f10.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f11.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f12.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f13.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f14.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f15.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f16.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f17.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f18.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f19.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f20.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f21.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f22.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f23.mgf',
        'Adult_Adrenalgland_Gel_Elite_49_f24.mgf'
    ]

    ADULT_ADRENALGLAND_GEL_VELOS_CRUX_FILES = [
        'Adult_Adrenalgland_Gel_Velos_2_f01.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f02.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f03.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f04.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f05.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f06.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f07.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f08.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f09.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f10.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f11.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f12.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f13.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f14.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f15.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f16.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f17.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f18.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f19.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f20.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f21.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f22.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f23.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f24.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f25.mgf',
        'Adult_Adrenalgland_Gel_Velos_2_f26.mgf'
    ]

    ADULT_ADRENALGLAND_BRP_VELOS_CRUX_FILES = [
        'Adult_Adrenalgland_bRP_Velos_1_f01.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f02.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f03.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f04.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f05.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f06.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f07.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f08.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f09.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f10.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f11.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f12.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f13.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f14.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f15.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f16.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f17.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f18.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f19.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f20.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f21.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f22.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f23.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f24.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f25.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f26.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f27.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f28.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f29.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f30.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f31.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f32.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f33.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f34.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f35.mgf',
        'Adult_Adrenalgland_bRP_Velos_1_f36.mgf'
    ]

    ADULT_HEART_BRP_ELITE_CRUX_FILES = [
        'Adult_Heart_bRP_Elite_81_f01.mgf',
        'Adult_Heart_bRP_Elite_81_f02.mgf',
        'Adult_Heart_bRP_Elite_81_f03.mgf',
        'Adult_Heart_bRP_Elite_81_f04.mgf',
        'Adult_Heart_bRP_Elite_81_f05.mgf',
        'Adult_Heart_bRP_Elite_81_f06.mgf',
        'Adult_Heart_bRP_Elite_81_f07.mgf',
        'Adult_Heart_bRP_Elite_81_f08.mgf',
        'Adult_Heart_bRP_Elite_81_f09.mgf',
        'Adult_Heart_bRP_Elite_81_f10.mgf',
        'Adult_Heart_bRP_Elite_81_f11.mgf',
        'Adult_Heart_bRP_Elite_81_f12.mgf',
        'Adult_Heart_bRP_Elite_81_f13.mgf',
        'Adult_Heart_bRP_Elite_81_f14.mgf',
        'Adult_Heart_bRP_Elite_81_f15.mgf',
        'Adult_Heart_bRP_Elite_81_f16.mgf',
        'Adult_Heart_bRP_Elite_81_f17.mgf',
        'Adult_Heart_bRP_Elite_81_f18.mgf',
        'Adult_Heart_bRP_Elite_81_f19.mgf',
        'Adult_Heart_bRP_Elite_81_f20.mgf',
        'Adult_Heart_bRP_Elite_81_f21.mgf',
        'Adult_Heart_bRP_Elite_81_f22.mgf',
        'Adult_Heart_bRP_Elite_81_f23.mgf',
        'Adult_Heart_bRP_Elite_81_f24.mgf'
    ]

    ADULT_HEART_BRP_VELOS_CRUX_FILES = [
        'Adult_Heart_bRP_Velos_6_f01.mgf',
        'Adult_Heart_bRP_Velos_6_f02.mgf',
        'Adult_Heart_bRP_Velos_6_f03.mgf',
        'Adult_Heart_bRP_Velos_6_f04.mgf',
        'Adult_Heart_bRP_Velos_6_f05.mgf',
        'Adult_Heart_bRP_Velos_6_f06.mgf',
        'Adult_Heart_bRP_Velos_6_f07.mgf',
        'Adult_Heart_bRP_Velos_6_f08.mgf',
        'Adult_Heart_bRP_Velos_6_f09.mgf',
        'Adult_Heart_bRP_Velos_6_f10.mgf',
        'Adult_Heart_bRP_Velos_6_f11.mgf',
        'Adult_Heart_bRP_Velos_6_f12.mgf',
        'Adult_Heart_bRP_Velos_6_f13.mgf',
        'Adult_Heart_bRP_Velos_6_f14.mgf',
        'Adult_Heart_bRP_Velos_6_f15.mgf',
        'Adult_Heart_bRP_Velos_6_f16.mgf',
        'Adult_Heart_bRP_Velos_6_f17.mgf',
        'Adult_Heart_bRP_Velos_6_f18.mgf',
        'Adult_Heart_bRP_Velos_6_f19.mgf',
        'Adult_Heart_bRP_Velos_6_f20.mgf',
        'Adult_Heart_bRP_Velos_6_f21.mgf',
        'Adult_Heart_bRP_Velos_6_f22.mgf',
        'Adult_Heart_bRP_Velos_6_f23.mgf',
        'Adult_Heart_bRP_Velos_6_f24.mgf',
        'Adult_Heart_bRP_Velos_6_f25.mgf',
        'Adult_Heart_bRP_Velos_6_f26.mgf'
    ]

    ADULT_PLATELETS_GEL_ELITE_CRUX_FILES = [
        'Adult_Platelets_Gel_Elite_48_f01.mgf',
        'Adult_Platelets_Gel_Elite_48_f02.mgf',
        'Adult_Platelets_Gel_Elite_48_f03.mgf',
        'Adult_Platelets_Gel_Elite_48_f04.mgf',
        'Adult_Platelets_Gel_Elite_48_f05.mgf',
        'Adult_Platelets_Gel_Elite_48_f06.mgf',
        'Adult_Platelets_Gel_Elite_48_f07.mgf',
        'Adult_Platelets_Gel_Elite_48_f08.mgf',
        'Adult_Platelets_Gel_Elite_48_f09.mgf',
        'Adult_Platelets_Gel_Elite_48_f10.mgf',
        'Adult_Platelets_Gel_Elite_48_f11.mgf',
        'Adult_Platelets_Gel_Elite_48_f12.mgf',
        'Adult_Platelets_Gel_Elite_48_f13.mgf',
        'Adult_Platelets_Gel_Elite_48_f14.mgf',
        'Adult_Platelets_Gel_Elite_48_f15.mgf',
        'Adult_Platelets_Gel_Elite_48_f16.mgf',
        'Adult_Platelets_Gel_Elite_48_f17.mgf',
        'Adult_Platelets_Gel_Elite_48_f18.mgf',
        'Adult_Platelets_Gel_Elite_48_f19.mgf',
        'Adult_Platelets_Gel_Elite_48_f20.mgf',
        'Adult_Platelets_Gel_Elite_48_f21.mgf',
        'Adult_Platelets_Gel_Elite_48_f22.mgf',
        'Adult_Platelets_Gel_Elite_48_f23.mgf',
        'Adult_Platelets_Gel_Elite_48_f24.mgf'
    ]

    ADULT_URINARYBLADDER_GEL_ELITE_CRUX_FILES = [
        'Adult_Urinarybladder_Gel_Elite_70_f01.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f02.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f03.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f04.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f05.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f06.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f07.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f08.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f09.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f10.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f11.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f12.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f13.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f14.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f15.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f16.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f17.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f18.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f19.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f20.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f21.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f22.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f23.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f24.mgf',
        'Adult_Urinarybladder_Gel_Elite_70_f25.mgf'
    ]

    FETAL_BRAIN_GEL_VELOS_CRUX_FILES = [
		"Fetal_Brain_Gel_Velos_16_f01.mgf",
		"Fetal_Brain_Gel_Velos_16_f02.mgf",
		"Fetal_Brain_Gel_Velos_16_f03.mgf",
		"Fetal_Brain_Gel_Velos_16_f04.mgf",
		"Fetal_Brain_Gel_Velos_16_f05.mgf",
		"Fetal_Brain_Gel_Velos_16_f06.mgf",
		"Fetal_Brain_Gel_Velos_16_f07.mgf",
		"Fetal_Brain_Gel_Velos_16_f08.mgf",
		"Fetal_Brain_Gel_Velos_16_f09.mgf",
		"Fetal_Brain_Gel_Velos_16_f10.mgf",
		"Fetal_Brain_Gel_Velos_16_f11.mgf",
		"Fetal_Brain_Gel_Velos_16_f12.mgf",
		"Fetal_Brain_Gel_Velos_16_f13.mgf",
		"Fetal_Brain_Gel_Velos_16_f14.mgf",
		"Fetal_Brain_Gel_Velos_16_f15.mgf",
		"Fetal_Brain_Gel_Velos_16_f16.mgf",
		"Fetal_Brain_Gel_Velos_16_f17.mgf",
		"Fetal_Brain_Gel_Velos_16_f18.mgf",
		"Fetal_Brain_Gel_Velos_16_f19.mgf",
		"Fetal_Brain_Gel_Velos_16_f20.mgf",
		"Fetal_Brain_Gel_Velos_16_f21.mgf",
		"Fetal_Brain_Gel_Velos_16_f22.mgf",
		"Fetal_Brain_Gel_Velos_16_f23.mgf",
		"Fetal_Brain_Gel_Velos_16_f24.mgf",
		"Fetal_Brain_Gel_Velos_16_f25.mgf",
		"Fetal_Brain_Gel_Velos_16_f26.mgf",
		"Fetal_Brain_Gel_Velos_16_f27.mgf",
		"Fetal_Brain_Gel_Velos_16_f28.mgf",
		"Fetal_Brain_Gel_Velos_16_f29.mgf"
    ]

    FETAL_LIVER_BRP_ELITE_CRUX_FILES = [
        'Fetal_Liver_bRP_Elite_22_f01.mgf',
        'Fetal_Liver_bRP_Elite_22_f02.mgf',
        'Fetal_Liver_bRP_Elite_22_f03.mgf',
        'Fetal_Liver_bRP_Elite_22_f04.mgf',
        'Fetal_Liver_bRP_Elite_22_f05.mgf',
        'Fetal_Liver_bRP_Elite_22_f06.mgf',
        'Fetal_Liver_bRP_Elite_22_f07.mgf',
        'Fetal_Liver_bRP_Elite_22_f08.mgf',
        'Fetal_Liver_bRP_Elite_22_f09.mgf',
        'Fetal_Liver_bRP_Elite_22_f10.mgf',
        'Fetal_Liver_bRP_Elite_22_f11.mgf',
        'Fetal_Liver_bRP_Elite_22_f12.mgf',
        'Fetal_Liver_bRP_Elite_22_f13.mgf',
        'Fetal_Liver_bRP_Elite_22_f14.mgf',
        'Fetal_Liver_bRP_Elite_22_f15.mgf',
        'Fetal_Liver_bRP_Elite_22_f16.mgf',
        'Fetal_Liver_bRP_Elite_22_f17.mgf',
        'Fetal_Liver_bRP_Elite_22_f18.mgf',
        'Fetal_Liver_bRP_Elite_22_f19.mgf',
        'Fetal_Liver_bRP_Elite_22_f20.mgf',
        'Fetal_Liver_bRP_Elite_22_f21.mgf',
        'Fetal_Liver_bRP_Elite_22_f22.mgf',
        'Fetal_Liver_bRP_Elite_22_f23.mgf',
        'Fetal_Liver_bRP_Elite_22_f24.mgf',
        'Fetal_Liver_bRP_Elite_22_f25.mgf',
        'Fetal_Liver_bRP_Elite_22_f26.mgf',
        'Fetal_Liver_bRP_Elite_22_f27.mgf',
        'Fetal_Liver_bRP_Elite_22_f28.mgf',
        'Fetal_Liver_bRP_Elite_22_f29.mgf',
        'Fetal_Liver_bRP_Elite_22_f30.mgf',
        'Fetal_Liver_bRP_Elite_22_f31.mgf',
        'Fetal_Liver_bRP_Elite_22_f32.mgf',
        'Fetal_Liver_bRP_Elite_22_f33.mgf',
        'Fetal_Liver_bRP_Elite_22_f34.mgf',
        'Fetal_Liver_bRP_Elite_22_f35.mgf'
    ]

    FETAL_OVARY_BRP_VELOS_CRUX_FILES = [
        'Fetal_Ovary_bRP_Velos_41_f01.mgf',
        'Fetal_Ovary_bRP_Velos_41_f02.mgf',
        'Fetal_Ovary_bRP_Velos_41_f03.mgf',
        'Fetal_Ovary_bRP_Velos_41_f04.mgf',
        'Fetal_Ovary_bRP_Velos_41_f05.mgf',
        'Fetal_Ovary_bRP_Velos_41_f06.mgf',
        'Fetal_Ovary_bRP_Velos_41_f07.mgf',
        'Fetal_Ovary_bRP_Velos_41_f08.mgf',
        'Fetal_Ovary_bRP_Velos_41_f09.mgf',
        'Fetal_Ovary_bRP_Velos_41_f10.mgf',
        'Fetal_Ovary_bRP_Velos_41_f11.mgf',
        'Fetal_Ovary_bRP_Velos_41_f12.mgf',
        'Fetal_Ovary_bRP_Velos_41_f13.mgf',
        'Fetal_Ovary_bRP_Velos_41_f14.mgf',
        'Fetal_Ovary_bRP_Velos_41_f15.mgf',
        'Fetal_Ovary_bRP_Velos_41_f16.mgf',
        'Fetal_Ovary_bRP_Velos_41_f17.mgf',
        'Fetal_Ovary_bRP_Velos_41_f18.mgf',
        'Fetal_Ovary_bRP_Velos_41_f19.mgf',
        'Fetal_Ovary_bRP_Velos_41_f20.mgf',
        'Fetal_Ovary_bRP_Velos_41_f21.mgf',
        'Fetal_Ovary_bRP_Velos_41_f22.mgf',
        'Fetal_Ovary_bRP_Velos_41_f23.mgf',
        'Fetal_Ovary_bRP_Velos_41_f24.mgf',
        'Fetal_Ovary_bRP_Velos_41_f25.mgf',
        'Fetal_Ovary_bRP_Velos_41_f26.mgf'
    ]

    FETAL_OVARY_BRP_ELITE_CRUX_FILES = [
        'Fetal_Ovary_bRP_Elite_25_f01.mgf',
        'Fetal_Ovary_bRP_Elite_25_f02.mgf',
        'Fetal_Ovary_bRP_Elite_25_f03.mgf',
        'Fetal_Ovary_bRP_Elite_25_f04.mgf',
        'Fetal_Ovary_bRP_Elite_25_f05.mgf',
        'Fetal_Ovary_bRP_Elite_25_f06.mgf',
        'Fetal_Ovary_bRP_Elite_25_f07.mgf',
        'Fetal_Ovary_bRP_Elite_25_f08.mgf',
        'Fetal_Ovary_bRP_Elite_25_f09.mgf',
        'Fetal_Ovary_bRP_Elite_25_f10.mgf',
        'Fetal_Ovary_bRP_Elite_25_f11.mgf',
        'Fetal_Ovary_bRP_Elite_25_f12.mgf',
        'Fetal_Ovary_bRP_Elite_25_f13.mgf',
        'Fetal_Ovary_bRP_Elite_25_f14.mgf',
        'Fetal_Ovary_bRP_Elite_25_f15.mgf',
        'Fetal_Ovary_bRP_Elite_25_f16.mgf',
        'Fetal_Ovary_bRP_Elite_25_f17.mgf',
        'Fetal_Ovary_bRP_Elite_25_f18.mgf',
        'Fetal_Ovary_bRP_Elite_25_f19.mgf',
        'Fetal_Ovary_bRP_Elite_25_f20.mgf',
        'Fetal_Ovary_bRP_Elite_25_f21.mgf',
        'Fetal_Ovary_bRP_Elite_25_f22.mgf',
        'Fetal_Ovary_bRP_Elite_25_f23.mgf',
        'Fetal_Ovary_bRP_Elite_25_f24.mgf'
    ]



    ADULT_BCELLS_BRP_ELITE_CRUX_FILES = [
        "Adult_Bcells_bRP_Elite_75_f01.mgf",
        "Adult_Bcells_bRP_Elite_75_f02.mgf",
        "Adult_Bcells_bRP_Elite_75_f03.mgf",
        "Adult_Bcells_bRP_Elite_75_f04.mgf",
        "Adult_Bcells_bRP_Elite_75_f05.mgf",
        "Adult_Bcells_bRP_Elite_75_f06.mgf",
        "Adult_Bcells_bRP_Elite_75_f07.mgf",
        "Adult_Bcells_bRP_Elite_75_f08.mgf",
        "Adult_Bcells_bRP_Elite_75_f09.mgf",
        "Adult_Bcells_bRP_Elite_75_f10.mgf",
        "Adult_Bcells_bRP_Elite_75_f11.mgf",
        "Adult_Bcells_bRP_Elite_75_f12.mgf",
        "Adult_Bcells_bRP_Elite_75_f13.mgf",
        "Adult_Bcells_bRP_Elite_75_f14.mgf",
        "Adult_Bcells_bRP_Elite_75_f15.mgf",
        "Adult_Bcells_bRP_Elite_75_f16.mgf",
        "Adult_Bcells_bRP_Elite_75_f17.mgf",
        "Adult_Bcells_bRP_Elite_75_f18.mgf",
        "Adult_Bcells_bRP_Elite_75_f19.mgf",
        "Adult_Bcells_bRP_Elite_75_f20.mgf",
        "Adult_Bcells_bRP_Elite_75_f21.mgf",
        "Adult_Bcells_bRP_Elite_75_f22.mgf",
        "Adult_Bcells_bRP_Elite_75_f23.mgf",
        "Adult_Bcells_bRP_Elite_75_f24.mgf"
    ]

    ADULT_BCELLS_BRP_VELOS_CRUX_FILES = [
        "Adult_Bcells_bRP_Velos_42_f01.mgf",
        "Adult_Bcells_bRP_Velos_42_f02.mgf",
        "Adult_Bcells_bRP_Velos_42_f03.mgf",
        "Adult_Bcells_bRP_Velos_42_f04.mgf",
        "Adult_Bcells_bRP_Velos_42_f05.mgf",
        "Adult_Bcells_bRP_Velos_42_f06.mgf",
        "Adult_Bcells_bRP_Velos_42_f07.mgf",
        "Adult_Bcells_bRP_Velos_42_f08.mgf",
        "Adult_Bcells_bRP_Velos_42_f09.mgf",
        "Adult_Bcells_bRP_Velos_42_f10.mgf",
        "Adult_Bcells_bRP_Velos_42_f11.mgf",
        "Adult_Bcells_bRP_Velos_42_f12.mgf",
        "Adult_Bcells_bRP_Velos_42_f13.mgf",
        "Adult_Bcells_bRP_Velos_42_f14.mgf",
        "Adult_Bcells_bRP_Velos_42_f15.mgf",
        "Adult_Bcells_bRP_Velos_42_f16.mgf",
        "Adult_Bcells_bRP_Velos_42_f17.mgf",
        "Adult_Bcells_bRP_Velos_42_f18.mgf",
        "Adult_Bcells_bRP_Velos_42_f19.mgf",
        "Adult_Bcells_bRP_Velos_42_f20.mgf",
        "Adult_Bcells_bRP_Velos_42_f21.mgf",
        "Adult_Bcells_bRP_Velos_42_f22.mgf",
        "Adult_Bcells_bRP_Velos_42_f23.mgf",
        "Adult_Bcells_bRP_Velos_42_f24.mgf",
        "Adult_Bcells_bRP_Velos_42_f25.mgf",
        "Adult_Bcells_bRP_Velos_42_f26.mgf",
        "Adult_Bcells_bRP_Velos_42_f27.mgf",
        "Adult_Bcells_bRP_Velos_42_f28.mgf",
        "Adult_Bcells_bRP_Velos_42_f29.mgf",
        "Adult_Bcells_bRP_Velos_42_f30.mgf",
        "Adult_Bcells_bRP_Velos_42_f31.mgf",
        "Adult_Bcells_bRP_Velos_42_f32.mgf"
    ]

    ADULT_BCELLS_GEL_ELITE_CRUX_FILES = [
        "Adult_Bcells_Gel_Elite_76_f01.mgf",
        "Adult_Bcells_Gel_Elite_76_f02.mgf",
        "Adult_Bcells_Gel_Elite_76_f03.mgf",
        "Adult_Bcells_Gel_Elite_76_f04.mgf",
        "Adult_Bcells_Gel_Elite_76_f05.mgf",
        "Adult_Bcells_Gel_Elite_76_f06.mgf",
        "Adult_Bcells_Gel_Elite_76_f07.mgf",
        "Adult_Bcells_Gel_Elite_76_f08.mgf",
        "Adult_Bcells_Gel_Elite_76_f09.mgf",
        "Adult_Bcells_Gel_Elite_76_f10.mgf",
        "Adult_Bcells_Gel_Elite_76_f11.mgf",
        "Adult_Bcells_Gel_Elite_76_f12.mgf",
        "Adult_Bcells_Gel_Elite_76_f13.mgf",
        "Adult_Bcells_Gel_Elite_76_f14.mgf",
        "Adult_Bcells_Gel_Elite_76_f15.mgf",
        "Adult_Bcells_Gel_Elite_76_f16.mgf",
        "Adult_Bcells_Gel_Elite_76_f17.mgf",
        "Adult_Bcells_Gel_Elite_76_f18.mgf",
        "Adult_Bcells_Gel_Elite_76_f19.mgf",
        "Adult_Bcells_Gel_Elite_76_f20.mgf",
        "Adult_Bcells_Gel_Elite_76_f21.mgf",
        "Adult_Bcells_Gel_Elite_76_f22.mgf",
        "Adult_Bcells_Gel_Elite_76_f23.mgf",
        "Adult_Bcells_Gel_Elite_76_f24.mgf"
    ]

    ADULT_CD4TCELLS_GEL_VELOS_CRUX_FILES = [
        "Adult_CD4Tcells_Gel_Velos_30_f01.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f02.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f03.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f04.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f05.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f06.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f07.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f08.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f09.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f10.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f11.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f12.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f13.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f14.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f15.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f16.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f17.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f18.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f19.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f20.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f21.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f22.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f23.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f24.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f25.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f26.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f27.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f28.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f29.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f30.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f31.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f32.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f33.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f34.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f35.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f36.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f37.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f38.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f39.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f40.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f41.mgf",
        "Adult_CD4Tcells_Gel_Velos_30_f42.mgf"
    ]

    ADULT_CD8TCELLS_GEL_ELITE_CRUX_FILES = [
        "Adult_CD8Tcells_Gel_Elite_44_f01.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f02.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f03.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f04.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f05.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f06.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f07.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f08.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f09.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f10.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f11.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f12.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f13.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f14.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f15.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f16.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f17.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f18.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f19.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f20.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f21.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f22.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f23.mgf",
        "Adult_CD8Tcells_Gel_Elite_44_f24.mgf"
    ]

    ADULT_COLON_GEL_ELITE_CRUX_FILES = [
        "Adult_Colon_Gel_Elite_51_f01.mgf",
        "Adult_Colon_Gel_Elite_51_f02.mgf",
        "Adult_Colon_Gel_Elite_51_f03.mgf",
        "Adult_Colon_Gel_Elite_51_f04.mgf",
        "Adult_Colon_Gel_Elite_51_f05.mgf",
        "Adult_Colon_Gel_Elite_51_f06.mgf",
        "Adult_Colon_Gel_Elite_51_f07.mgf",
        "Adult_Colon_Gel_Elite_51_f08.mgf",
        "Adult_Colon_Gel_Elite_51_f09.mgf",
        "Adult_Colon_Gel_Elite_51_f10.mgf",
        "Adult_Colon_Gel_Elite_51_f11.mgf",
        "Adult_Colon_Gel_Elite_51_f12.mgf",
        "Adult_Colon_Gel_Elite_51_f13.mgf",
        "Adult_Colon_Gel_Elite_51_f14.mgf",
        "Adult_Colon_Gel_Elite_51_f15.mgf",
        "Adult_Colon_Gel_Elite_51_f16.mgf",
        "Adult_Colon_Gel_Elite_51_f17.mgf",
        "Adult_Colon_Gel_Elite_51_f18.mgf",
        "Adult_Colon_Gel_Elite_51_f19.mgf",
        "Adult_Colon_Gel_Elite_51_f20.mgf",
        "Adult_Colon_Gel_Elite_51_f21.mgf",
        "Adult_Colon_Gel_Elite_51_f22.mgf",
        "Adult_Colon_Gel_Elite_51_f23.mgf",
        "Adult_Colon_Gel_Elite_51_f24.mgf"
    ]

    ADULT_ESOPHAGUS_GEL_VELOS_CRUX_FILES = [
        "Adult_Esophagus_Gel_Velos_4_f01.mgf",
        "Adult_Esophagus_Gel_Velos_4_f02.mgf",
        "Adult_Esophagus_Gel_Velos_4_f03.mgf",
        "Adult_Esophagus_Gel_Velos_4_f04.mgf",
        "Adult_Esophagus_Gel_Velos_4_f05.mgf",
        "Adult_Esophagus_Gel_Velos_4_f06.mgf",
        "Adult_Esophagus_Gel_Velos_4_f07.mgf",
        "Adult_Esophagus_Gel_Velos_4_f08.mgf",
        "Adult_Esophagus_Gel_Velos_4_f09.mgf",
        "Adult_Esophagus_Gel_Velos_4_f10.mgf",
        "Adult_Esophagus_Gel_Velos_4_f11.mgf",
        "Adult_Esophagus_Gel_Velos_4_f12.mgf",
        "Adult_Esophagus_Gel_Velos_4_f13.mgf",
        "Adult_Esophagus_Gel_Velos_4_f14.mgf",
        "Adult_Esophagus_Gel_Velos_4_f15.mgf",
        "Adult_Esophagus_Gel_Velos_4_f16.mgf",
        "Adult_Esophagus_Gel_Velos_4_f17.mgf",
        "Adult_Esophagus_Gel_Velos_4_f18.mgf",
        "Adult_Esophagus_Gel_Velos_4_f19.mgf",
        "Adult_Esophagus_Gel_Velos_4_f20.mgf",
        "Adult_Esophagus_Gel_Velos_4_f21.mgf",
        "Adult_Esophagus_Gel_Velos_4_f22.mgf",
        "Adult_Esophagus_Gel_Velos_4_f23.mgf",
        "Adult_Esophagus_Gel_Velos_4_f24.mgf"
    ]

    ADULT_NKCELLS_BRP_ELITE_CRUX_FILES = [
        "Adult_NKcells_bRP_Elite_34_f01.mgf",
        "Adult_NKcells_bRP_Elite_34_f02.mgf",
        "Adult_NKcells_bRP_Elite_34_f03.mgf",
        "Adult_NKcells_bRP_Elite_34_f04.mgf",
        "Adult_NKcells_bRP_Elite_34_f05.mgf",
        "Adult_NKcells_bRP_Elite_34_f06.mgf",
        "Adult_NKcells_bRP_Elite_34_f07.mgf",
        "Adult_NKcells_bRP_Elite_34_f08.mgf",
        "Adult_NKcells_bRP_Elite_34_f09.mgf",
        "Adult_NKcells_bRP_Elite_34_f10.mgf",
        "Adult_NKcells_bRP_Elite_34_f11.mgf",
        "Adult_NKcells_bRP_Elite_34_f12.mgf",
        "Adult_NKcells_bRP_Elite_34_f13.mgf",
        "Adult_NKcells_bRP_Elite_34_f14.mgf",
        "Adult_NKcells_bRP_Elite_34_f15.mgf",
        "Adult_NKcells_bRP_Elite_34_f16.mgf",
        "Adult_NKcells_bRP_Elite_34_f17.mgf",
        "Adult_NKcells_bRP_Elite_34_f18.mgf",
        "Adult_NKcells_bRP_Elite_34_f19.mgf",
        "Adult_NKcells_bRP_Elite_34_f20.mgf",
        "Adult_NKcells_bRP_Elite_34_f21.mgf",
        "Adult_NKcells_bRP_Elite_34_f22.mgf",
        "Adult_NKcells_bRP_Elite_34_f23.mgf",
        "Adult_NKcells_bRP_Elite_34_f24.mgf",
        "Adult_NKcells_bRP_Elite_34_f25.mgf",
        "Adult_NKcells_bRP_Elite_34_f26.mgf",
        "Adult_NKcells_bRP_Elite_34_f27.mgf",
        "Adult_NKcells_bRP_Elite_34_f28.mgf"
    ]

    ADULT_NKCELLS_BRP_VELOS_CRUX_FILES = [
        "Adult_NKcells_bRP_Velos_46_f01.mgf",
        "Adult_NKcells_bRP_Velos_46_f02.mgf",
        "Adult_NKcells_bRP_Velos_46_f03.mgf",
        "Adult_NKcells_bRP_Velos_46_f04.mgf",
        "Adult_NKcells_bRP_Velos_46_f05.mgf",
        "Adult_NKcells_bRP_Velos_46_f06.mgf",
        "Adult_NKcells_bRP_Velos_46_f07.mgf",
        "Adult_NKcells_bRP_Velos_46_f08.mgf",
        "Adult_NKcells_bRP_Velos_46_f09.mgf",
        "Adult_NKcells_bRP_Velos_46_f10.mgf",
        "Adult_NKcells_bRP_Velos_46_f11.mgf",
        "Adult_NKcells_bRP_Velos_46_f12.mgf",
        "Adult_NKcells_bRP_Velos_46_f13.mgf",
        "Adult_NKcells_bRP_Velos_46_f14.mgf",
        "Adult_NKcells_bRP_Velos_46_f15.mgf",
        "Adult_NKcells_bRP_Velos_46_f16.mgf",
        "Adult_NKcells_bRP_Velos_46_f17.mgf",
        "Adult_NKcells_bRP_Velos_46_f18.mgf",
        "Adult_NKcells_bRP_Velos_46_f19.mgf",
        "Adult_NKcells_bRP_Velos_46_f20.mgf",
        "Adult_NKcells_bRP_Velos_46_f21.mgf",
        "Adult_NKcells_bRP_Velos_46_f22.mgf",
        "Adult_NKcells_bRP_Velos_46_f23.mgf",
        "Adult_NKcells_bRP_Velos_46_f24.mgf"
    ]

    ADULT_PANCREAS_BRP_ELITE_CRUX_FILES = [
        "Adult_Pancreas_bRP_Elite_59_f01.mgf",
        "Adult_Pancreas_bRP_Elite_59_f02.mgf",
        "Adult_Pancreas_bRP_Elite_59_f03.mgf",
        "Adult_Pancreas_bRP_Elite_59_f04.mgf",
        "Adult_Pancreas_bRP_Elite_59_f05.mgf",
        "Adult_Pancreas_bRP_Elite_59_f06.mgf",
        "Adult_Pancreas_bRP_Elite_59_f07.mgf",
        "Adult_Pancreas_bRP_Elite_59_f08.mgf",
        "Adult_Pancreas_bRP_Elite_59_f09.mgf",
        "Adult_Pancreas_bRP_Elite_59_f10.mgf",
        "Adult_Pancreas_bRP_Elite_59_f11.mgf",
        "Adult_Pancreas_bRP_Elite_59_f12.mgf",
        "Adult_Pancreas_bRP_Elite_59_f13.mgf",
        "Adult_Pancreas_bRP_Elite_59_f14.mgf",
        "Adult_Pancreas_bRP_Elite_59_f15.mgf",
        "Adult_Pancreas_bRP_Elite_59_f16.mgf",
        "Adult_Pancreas_bRP_Elite_59_f17.mgf",
        "Adult_Pancreas_bRP_Elite_59_f18.mgf",
        "Adult_Pancreas_bRP_Elite_59_f19.mgf",
        "Adult_Pancreas_bRP_Elite_59_f20.mgf",
        "Adult_Pancreas_bRP_Elite_59_f21.mgf",
        "Adult_Pancreas_bRP_Elite_59_f22.mgf",
        "Adult_Pancreas_bRP_Elite_59_f23.mgf",
        "Adult_Pancreas_bRP_Elite_59_f24.mgf"
    ]

    FETAL_GUT_GEL_VELOS_CRUX_FILES = [
        "Fetal_Gut_Gel_Velos_72_f01.mgf",
        "Fetal_Gut_Gel_Velos_72_f02.mgf",
        "Fetal_Gut_Gel_Velos_72_f03.mgf",
        "Fetal_Gut_Gel_Velos_72_f04.mgf",
        "Fetal_Gut_Gel_Velos_72_f05.mgf",
        "Fetal_Gut_Gel_Velos_72_f06.mgf",
        "Fetal_Gut_Gel_Velos_72_f07.mgf",
        "Fetal_Gut_Gel_Velos_72_f08.mgf",
        "Fetal_Gut_Gel_Velos_72_f09.mgf",
        "Fetal_Gut_Gel_Velos_72_f10.mgf",
        "Fetal_Gut_Gel_Velos_72_f11.mgf",
        "Fetal_Gut_Gel_Velos_72_f12.mgf",
        "Fetal_Gut_Gel_Velos_72_f13.mgf",
        "Fetal_Gut_Gel_Velos_72_f14.mgf",
        "Fetal_Gut_Gel_Velos_72_f15.mgf",
        "Fetal_Gut_Gel_Velos_72_f16.mgf",
        "Fetal_Gut_Gel_Velos_72_f17.mgf",
        "Fetal_Gut_Gel_Velos_72_f18.mgf",
        "Fetal_Gut_Gel_Velos_72_f19.mgf",
        "Fetal_Gut_Gel_Velos_72_f20.mgf",
        "Fetal_Gut_Gel_Velos_72_f21.mgf",
        "Fetal_Gut_Gel_Velos_72_f22.mgf",
        "Fetal_Gut_Gel_Velos_72_f23.mgf",
        "Fetal_Gut_Gel_Velos_72_f24.mgf"
    ]

    FETAL_LIVER_BRP_ELITE_23_CRUX_FILES = [
        "Fetal_Liver_bRP_Elite_23_f01.mgf",
        "Fetal_Liver_bRP_Elite_23_f02.mgf",
        "Fetal_Liver_bRP_Elite_23_f03.mgf",
        "Fetal_Liver_bRP_Elite_23_f04.mgf",
        "Fetal_Liver_bRP_Elite_23_f05.mgf",
        "Fetal_Liver_bRP_Elite_23_f06.mgf",
        "Fetal_Liver_bRP_Elite_23_f07.mgf",
        "Fetal_Liver_bRP_Elite_23_f08.mgf",
        "Fetal_Liver_bRP_Elite_23_f09.mgf",
        "Fetal_Liver_bRP_Elite_23_f10.mgf",
        "Fetal_Liver_bRP_Elite_23_f11.mgf",
        "Fetal_Liver_bRP_Elite_23_f12.mgf",
        "Fetal_Liver_bRP_Elite_23_f13.mgf",
        "Fetal_Liver_bRP_Elite_23_f14.mgf",
        "Fetal_Liver_bRP_Elite_23_f15.mgf",
        "Fetal_Liver_bRP_Elite_23_f16.mgf",
        "Fetal_Liver_bRP_Elite_23_f17.mgf",
        "Fetal_Liver_bRP_Elite_23_f18.mgf",
        "Fetal_Liver_bRP_Elite_23_f19.mgf",
        "Fetal_Liver_bRP_Elite_23_f20.mgf",
        "Fetal_Liver_bRP_Elite_23_f21.mgf",
        "Fetal_Liver_bRP_Elite_23_f22.mgf",
        "Fetal_Liver_bRP_Elite_23_f23.mgf",
        "Fetal_Liver_bRP_Elite_23_f24.mgf"
    ]

    FETAL_LIVER_GEL_VELOS_CRUX_FILES = [
        "Fetal_Liver_Gel_Velos_24_f01.mgf",
        "Fetal_Liver_Gel_Velos_24_f02.mgf",
        "Fetal_Liver_Gel_Velos_24_f03.mgf",
        "Fetal_Liver_Gel_Velos_24_f04.mgf",
        "Fetal_Liver_Gel_Velos_24_f05.mgf",
        "Fetal_Liver_Gel_Velos_24_f06.mgf",
        "Fetal_Liver_Gel_Velos_24_f07.mgf",
        "Fetal_Liver_Gel_Velos_24_f08.mgf",
        "Fetal_Liver_Gel_Velos_24_f09.mgf",
        "Fetal_Liver_Gel_Velos_24_f10.mgf",
        "Fetal_Liver_Gel_Velos_24_f11.mgf",
        "Fetal_Liver_Gel_Velos_24_f12.mgf",
        "Fetal_Liver_Gel_Velos_24_f13.mgf",
        "Fetal_Liver_Gel_Velos_24_f14.mgf",
        "Fetal_Liver_Gel_Velos_24_f15.mgf",
        "Fetal_Liver_Gel_Velos_24_f16.mgf",
        "Fetal_Liver_Gel_Velos_24_f17.mgf",
        "Fetal_Liver_Gel_Velos_24_f18.mgf",
        "Fetal_Liver_Gel_Velos_24_f19.mgf",
        "Fetal_Liver_Gel_Velos_24_f20.mgf",
        "Fetal_Liver_Gel_Velos_24_f21.mgf",
        "Fetal_Liver_Gel_Velos_24_f22.mgf",
        "Fetal_Liver_Gel_Velos_24_f23.mgf",
        "Fetal_Liver_Gel_Velos_24_f24.mgf",
        "Fetal_Liver_Gel_Velos_24_f25.mgf",
        "Fetal_Liver_Gel_Velos_24_f26.mgf",
        "Fetal_Liver_Gel_Velos_24_f27.mgf"
    ]

    FETAL_TESTIS_BRP_ELITE_CRUX_FILES = [
        "Fetal_Testis_bRP_Elite_26_f01.mgf",
        "Fetal_Testis_bRP_Elite_26_f02.mgf",
        "Fetal_Testis_bRP_Elite_26_f03.mgf",
        "Fetal_Testis_bRP_Elite_26_f04.mgf",
        "Fetal_Testis_bRP_Elite_26_f05.mgf",
        "Fetal_Testis_bRP_Elite_26_f06.mgf",
        "Fetal_Testis_bRP_Elite_26_f07.mgf",
        "Fetal_Testis_bRP_Elite_26_f08.mgf",
        "Fetal_Testis_bRP_Elite_26_f09.mgf",
        "Fetal_Testis_bRP_Elite_26_f10.mgf",
        "Fetal_Testis_bRP_Elite_26_f11.mgf",
        "Fetal_Testis_bRP_Elite_26_f12.mgf",
        "Fetal_Testis_bRP_Elite_26_f13.mgf",
        "Fetal_Testis_bRP_Elite_26_f14.mgf",
        "Fetal_Testis_bRP_Elite_26_f15.mgf",
        "Fetal_Testis_bRP_Elite_26_f16.mgf",
        "Fetal_Testis_bRP_Elite_26_f17.mgf",
        "Fetal_Testis_bRP_Elite_26_f18.mgf",
        "Fetal_Testis_bRP_Elite_26_f19.mgf",
        "Fetal_Testis_bRP_Elite_26_f20.mgf",
        "Fetal_Testis_bRP_Elite_26_f21.mgf",
        "Fetal_Testis_bRP_Elite_26_f22.mgf",
        "Fetal_Testis_bRP_Elite_26_f23.mgf",
        "Fetal_Testis_bRP_Elite_26_f24.mgf",
        "Fetal_Testis_bRP_Elite_26_f25.mgf",
        "Fetal_Testis_bRP_Elite_26_f26.mgf",
        "Fetal_Testis_bRP_Elite_26_f27.mgf",
        "Fetal_Testis_bRP_Elite_26_f28.mgf",
        "Fetal_Testis_bRP_Elite_26_f29.mgf"
    ]



    #
    # This dictionary maps the identifications file into the corresponding experimental .mgf spectra.
    # 
    # Multiple identifications files for the same experiment indicates different confidence levels.
    #

    MATCHES_TO_CRUX_FILES_LIST = {
        "Adult_Adrenalgland_Gel_Elite_q_lt_0.01_identifications.tsv" : ADULT_ADRENALGLAND_GEL_ELITE_CRUX_FILES,
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.01_identifications.tsv" : ADULT_ADRENALGLAND_GEL_VELOS_CRUX_FILES,
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.01_identifications.tsv" : ADULT_ADRENALGLAND_BRP_VELOS_CRUX_FILES,
        "Adult_Heart_bRP_Velos_q_lt_0.01_identifications.tsv" : ADULT_HEART_BRP_VELOS_CRUX_FILES,
        "Adult_Platelets_Gel_Elite_q_lt_0.01_identifications.tsv" : ADULT_PLATELETS_GEL_ELITE_CRUX_FILES,
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.01_identifications.tsv" : ADULT_URINARYBLADDER_GEL_ELITE_CRUX_FILES,
        "Fetal_Brain_Gel_Velos_q_lt_0.01_identifications.tsv" : FETAL_BRAIN_GEL_VELOS_CRUX_FILES,
        "Fetal_Ovary_bRP_Velos_q_lt_0.01_identifications.tsv" : FETAL_OVARY_BRP_VELOS_CRUX_FILES,
        "Fetal_Ovary_bRP_Elite_q_lt_0.01_identifications.tsv" : FETAL_OVARY_BRP_ELITE_CRUX_FILES,

        "Adult_Bcells_bRP_Elite_q_lt_0.01_identifications.tsv" : ADULT_BCELLS_BRP_ELITE_CRUX_FILES,
        "Adult_Bcells_bRP_Velos_q_lt_0.01_identifications.tsv" : ADULT_BCELLS_BRP_VELOS_CRUX_FILES,
        "Adult_Bcells_Gel_Elite_q_lt_0.01_identifications.tsv" : ADULT_BCELLS_GEL_ELITE_CRUX_FILES,
        "Adult_CD4Tcells_Gel_Velos_q_lt_0.01_identifications.tsv" : ADULT_CD4TCELLS_GEL_VELOS_CRUX_FILES,
        "Adult_CD8Tcells_Gel_Elite_q_lt_0.01_identifications.tsv" : ADULT_CD8TCELLS_GEL_ELITE_CRUX_FILES,
        "Adult_Colon_Gel_Elite_q_lt_0.01_identifications.tsv" : ADULT_COLON_GEL_ELITE_CRUX_FILES,
        "Adult_Esophagus_Gel_Velos_q_lt_0.01_identifications.tsv" : ADULT_ESOPHAGUS_GEL_VELOS_CRUX_FILES,
        "Adult_NKcells_bRP_Elite_q_lt_0.01_identifications.tsv" : ADULT_NKCELLS_BRP_ELITE_CRUX_FILES,
        "Adult_NKcells_bRP_Velos_q_lt_0.01_identifications.tsv" : ADULT_NKCELLS_BRP_VELOS_CRUX_FILES,
        "Adult_Pancreas_bRP_Elite_q_lt_0.01_identifications.tsv" : ADULT_PANCREAS_BRP_ELITE_CRUX_FILES,
        "Fetal_Gut_Gel_Velos_q_lt_0.01_identifications.tsv" : FETAL_GUT_GEL_VELOS_CRUX_FILES,
        "Fetal_Liver_bRP_Elite_23_q_lt_0.01_identifications.tsv" : FETAL_LIVER_BRP_ELITE_23_CRUX_FILES,
        "Fetal_Liver_Gel_Velos_q_lt_0.01_identifications.tsv" : FETAL_LIVER_GEL_VELOS_CRUX_FILES,
        "Fetal_Testis_bRP_Elite_q_lt_0.01_identifications.tsv" : FETAL_TESTIS_BRP_ELITE_CRUX_FILES,

        "Adult_Adrenalgland_Gel_Elite_q_lt_0.001_identifications.tsv" : ADULT_ADRENALGLAND_GEL_ELITE_CRUX_FILES,
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.001_identifications.tsv" : ADULT_ADRENALGLAND_GEL_VELOS_CRUX_FILES,
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.001_identifications.tsv" : ADULT_ADRENALGLAND_BRP_VELOS_CRUX_FILES,
        "Adult_Heart_bRP_Velos_q_lt_0.001_identifications.tsv" : ADULT_HEART_BRP_VELOS_CRUX_FILES,
        "Adult_Platelets_Gel_Elite_q_lt_0.001_identifications.tsv" : ADULT_PLATELETS_GEL_ELITE_CRUX_FILES,
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.001_identifications.tsv" : ADULT_URINARYBLADDER_GEL_ELITE_CRUX_FILES,
        "Fetal_Brain_Gel_Velos_q_lt_0.001_identifications.tsv" : FETAL_BRAIN_GEL_VELOS_CRUX_FILES,
        "Fetal_Ovary_bRP_Velos_q_lt_0.001_identifications.tsv" : FETAL_OVARY_BRP_VELOS_CRUX_FILES,
        "Fetal_Ovary_bRP_Elite_q_lt_0.001_identifications.tsv" : FETAL_OVARY_BRP_ELITE_CRUX_FILES,

        "Adult_Bcells_bRP_Elite_q_lt_0.001_identifications.tsv" : ADULT_BCELLS_BRP_ELITE_CRUX_FILES,
        "Adult_Bcells_bRP_Velos_q_lt_0.001_identifications.tsv" : ADULT_BCELLS_BRP_VELOS_CRUX_FILES,
        "Adult_Bcells_Gel_Elite_q_lt_0.001_identifications.tsv" : ADULT_BCELLS_GEL_ELITE_CRUX_FILES,
        "Adult_CD4Tcells_Gel_Velos_q_lt_0.001_identifications.tsv" : ADULT_CD4TCELLS_GEL_VELOS_CRUX_FILES,
        "Adult_CD8Tcells_Gel_Elite_q_lt_0.001_identifications.tsv" : ADULT_CD8TCELLS_GEL_ELITE_CRUX_FILES,
        "Adult_Colon_Gel_Elite_q_lt_0.001_identifications.tsv" : ADULT_COLON_GEL_ELITE_CRUX_FILES,
        "Adult_Esophagus_Gel_Velos_q_lt_0.001_identifications.tsv" : ADULT_ESOPHAGUS_GEL_VELOS_CRUX_FILES,
        "Adult_NKcells_bRP_Elite_q_lt_0.001_identifications.tsv" : ADULT_NKCELLS_BRP_ELITE_CRUX_FILES,
        "Adult_NKcells_bRP_Velos_q_lt_0.001_identifications.tsv" : ADULT_NKCELLS_BRP_VELOS_CRUX_FILES,
        "Adult_Pancreas_bRP_Elite_q_lt_0.001_identifications.tsv" : ADULT_PANCREAS_BRP_ELITE_CRUX_FILES,
        "Fetal_Gut_Gel_Velos_q_lt_0.001_identifications.tsv" : FETAL_GUT_GEL_VELOS_CRUX_FILES,
        "Fetal_Liver_bRP_Elite_23_q_lt_0.001_identifications.tsv" : FETAL_LIVER_BRP_ELITE_23_CRUX_FILES,
        "Fetal_Liver_Gel_Velos_q_lt_0.001_identifications.tsv" : FETAL_LIVER_GEL_VELOS_CRUX_FILES,
        "Fetal_Testis_bRP_Elite_q_lt_0.001_identifications.tsv" : FETAL_TESTIS_BRP_ELITE_CRUX_FILES
    }



    def __init__(self, identificationsFilename = None, spectraFilename = None, cruxIdentifications = False):
    
        self.identificationsFilename = identificationsFilename
        self.spectraFilename = spectraFilename
        self.cruxIdentifications = cruxIdentifications
    
        self.totalSpectra = SpectraFound(False, 'sequences')
    
    
    def load_identifications(self, verbose = False, filteredFilesList = None):

        #
        # First check if the spectra has already been loaded
        #
        
        self.totalSpectra.load_spectra(self.spectraFilename)
        
        if self.totalSpectra.spectra: 
            return
        
        print('Loading file: {}. dir:{}'.format(self.identificationsFilename, os.getcwd()))

        if self.cruxIdentifications:
            self.uniqueCombination = pd.read_csv(self.identificationsFilename, sep='\t')
        else:
            matches_file = pd.read_csv(self.identificationsFilename)

            if verbose:
                print(matches_file)

            print('Number of unique sequences found: {}'.format(len(matches_file['Sequence'].unique())))

            #
            # Inject new columns to hold the scan sequence within the file and the file name, recovered from the 
            # "Spectrum Title" information.
            #

            matches_file['File Sequence'] = \
                matches_file['Spectrum Title'].str.split("_", expand = True).iloc[:, 3].str.zfill(MGF.SCAN_SEQUENCE_MAX_DIGITS)
            
            matches_file['File'] = matches_file['Spectrum Title'].str.split("_", expand = True).iloc[:, 2]

            ordered = matches_file.sort_values(['File', 'File Sequence'])


            #
            # Select only unique "Sequence" + "First Scan" combination:
            #
            # - Same spectrum can contain different sequences
            # - Different sequences can be read in different scans

            duplicatesIndication = ordered.duplicated(['Spectrum Title', 'Sequence', 'First Scan'])

            self.uniqueCombination = ordered[~duplicatesIndication]
            print('Unique combinations found: {}'.format(self.uniqueCombination.shape))

            # print('Unique combinations file {}: {}'.format('b01', 
            #                                                self.uniqueCombination[self.uniqueCombination['File'] == 'b01'].shape))

            if filteredFilesList:
                self.uniqueCombination = self.uniqueCombination[self.uniqueCombination['File'].isin(filteredFilesList)]


    def read_spectra_crux(self, spectraParser, storeUnrecognized = True):
        currentFileName      = ''
        currentScanFileIndex = ''

        currentFile  = None
        lastScan     = None

        startTime = time.time()

        spectraFiles = PXD000561.MATCHES_TO_CRUX_FILES_LIST[self.identificationsFilename]

        for index, row in self.uniqueCombination.iterrows():
            try:
                if (spectraFiles[row['file_idx']] != currentFileName):
                    if (currentFile != None):

                        if storeUnrecognized:
                            #
                            # Dumps the rest of current file as unrecognized spectra
                            #

                            spectraParser.read_spectrum(currentFile, str(currentScanFileIndex) + '_', 
                                                        MGF.SCAN_SEQUENCE_OVER_LIMIT, 
                                                        '', 
                                                        self.totalSpectra, useScanIndex = True)

                            print('File {}. Processing time of {} seconds'.format(currentFile.name, 
                                                                                time.time() - startTime))

                        currentFile.close()

                        # break

                    currentScanFileIndex = row['file_idx']
                    currentFileName = spectraFiles[currentScanFileIndex]

                    print('Will open file \"{}\"'.format(currentFileName))

                    currentFile = open(currentFileName, 'r')
                    lastScan    = None

                _, lastScan, _ = spectraParser.read_spectrum(currentFile, 
                                                             str(currentScanFileIndex) + '_', 
                                                             row['scan'], 
                                                             row['sequence'], 
                                                             self.totalSpectra, 
                                                             currentScan = lastScan,
                                                             storeUnrecognized = storeUnrecognized,
                                                             useScanIndex = True)
            except KeyError:
                Logger()("- Identification file key {} has not corresponding spectra file.")

        Logger()('{} Total processing time {} seconds'.format(self.identificationsFilename, time.time() - startTime))
        
        self.totalSpectra.save_spectra(self.spectraFilename)


    def read_spectra_msf(self, spectraParser, storeUnrecognized = True):
        currentFileName           = ''
        currentScanFileNamePrefix = ''

        currentFile  = None
        lastScan     = None

        startTime = time.time()

        spectraFiles = PXD000561.MATCHES_TO_FILES_LIST[self.identificationsFilename]

        for index, row in self.uniqueCombination.iterrows():
            try:
                if (spectraFiles[row['File']] != currentFileName):
                    if (currentFile != None):

                        if storeUnrecognized:
                            #
                            # Dumps the rest of current file as unrecognized spectra
                            #

                            spectraParser.read_spectrum(currentFile, currentFileNamePrefix + '_', 
                                                        MGF.SCAN_SEQUENCE_OVER_LIMIT, 
                                                        '', 
                                                        self.totalSpectra)

                            print('File {}. Processing time of {} seconds'.format(currentFile.name, 
                                                                                time.time() - startTime))

                        currentFile.close()

                        # break

                    currentFileNamePrefix = row['File']
                    currentFileName = spectraFiles[currentFileNamePrefix]

                    print('Will open file \"{}\"'.format(currentFileName))

                    currentFile = open(currentFileName, 'r')
                    lastScan    = None

                _, lastScan, _ = spectraParser.read_spectrum(currentFile, 
                                                             currentFileNamePrefix + '_', 
                                                             row['First Scan'], 
                                                             row['Sequence'], 
                                                             self.totalSpectra, 
                                                             currentScan = lastScan,
                                                             storeUnrecognized = storeUnrecognized)
            except KeyError:
                Logger()("- Identification file key {} has not corresponding spectra file.")

        Logger()('{} Total processing time {} seconds'.format(self.identificationsFilename, time.time() - startTime))
        
        self.totalSpectra.save_spectra(self.spectraFilename)


    def read_spectra(self, spectraParser, storeUnrecognized = True):
        if self.cruxIdentifications:
            self.read_spectra_crux(spectraParser, storeUnrecognized)
        else:
            self.read_spectra_msf(spectraParser, storeUnrecognized)




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

        #
        # Removing this file since Crux has not found any identification here
        #
        # 'Fetal_Liver_bRP_Elite_22_f28.mgf',
        
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
        "Adult_CD4Tcells_Gel_Velos_30_f41.mgf"
        #
        # Removing this file since it has no identifications
        #
        #, "Adult_CD4Tcells_Gel_Velos_30_f42.mgf"
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
    # New sequence of files completing all the experiments
    #

    ADULT_CD4TCELLS_BRP_ELITE_28_FILES = [
        "Adult_CD4Tcells_bRP_Elite_28_f01.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f02.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f03.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f04.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f05.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f06.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f07.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f08.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f09.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f10.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f11.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f12.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f13.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f14.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f15.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f16.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f17.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f18.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f19.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f20.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f21.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f22.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f23.mgf",
        "Adult_CD4Tcells_bRP_Elite_28_f24.mgf"
    ]

    ADULT_CD4TCELLS_BRP_VELOS_29_FILES = [
        "Adult_CD4Tcells_bRP_Velos_29_f01.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f02.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f03.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f04.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f05.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f06.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f07.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f08.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f09.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f10.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f11.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f12.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f13.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f14.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f15.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f16.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f17.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f18.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f19.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f20.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f21.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f22.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f23.mgf",
        "Adult_CD4Tcells_bRP_Velos_29_f24.mgf"
    ]

    ADULT_CD8TCELLS_GEL_VELOS_45_FILES = [
        "Adult_CD8Tcells_Gel_Velos_45_f01.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f02.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f03.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f05.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f06.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f07.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f08.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f09.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f10.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f11.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f12.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f13.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f14.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f15.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f16.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f17.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f18.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f19.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f20.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f21.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f22.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f23.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f24.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f25.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f26.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f27.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f28.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f29.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f30.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f31.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f32.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f33.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f34.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f35.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f36.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f37.mgf",
        "Adult_CD8Tcells_Gel_Velos_45_f38.mgf"
    ]

    ADULT_CD8TCELLS_BRP_ELITE_77_FILES = [
        "Adult_CD8Tcells_bRP_Elite_77_f01.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f02.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f03.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f04.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f05.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f06.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f07.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f08.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f09.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f10.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f11.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f12.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f13.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f14.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f15.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f16.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f17.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f18.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f19.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f20.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f21.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f22.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f23.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f24.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f25.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f26.mgf",
        "Adult_CD8Tcells_bRP_Elite_77_f27.mgf"
    ]

    ADULT_CD8TCELLS_BRP_VELOS_43_FILES = [
        "Adult_CD8Tcells_bRP_Velos_43_f02.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f03.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f04.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f05.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f06.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f07.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f08.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f09.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f10.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f11.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f12.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f13.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f14.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f15.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f16.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f17.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f18.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f19.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f20.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f21.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f22.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f23.mgf",
        "Adult_CD8Tcells_bRP_Velos_43_f24.mgf"
    ]

    ADULT_COLON_BRP_ELITE_50_FILES = [
        "Adult_Colon_bRP_Elite_50_f01.mgf",
        "Adult_Colon_bRP_Elite_50_f02.mgf",
        "Adult_Colon_bRP_Elite_50_f03.mgf",
        "Adult_Colon_bRP_Elite_50_f04.mgf",
        "Adult_Colon_bRP_Elite_50_f05.mgf",
        "Adult_Colon_bRP_Elite_50_f06.mgf",
        "Adult_Colon_bRP_Elite_50_f07.mgf",
        "Adult_Colon_bRP_Elite_50_f08.mgf",
        "Adult_Colon_bRP_Elite_50_f10.mgf",
        "Adult_Colon_bRP_Elite_50_f11.mgf",
        "Adult_Colon_bRP_Elite_50_f12.mgf",
        "Adult_Colon_bRP_Elite_50_f13.mgf",
        "Adult_Colon_bRP_Elite_50_f14.mgf",
        "Adult_Colon_bRP_Elite_50_f15.mgf",
        "Adult_Colon_bRP_Elite_50_f16.mgf",
        "Adult_Colon_bRP_Elite_50_f17.mgf",
        "Adult_Colon_bRP_Elite_50_f18.mgf",
        "Adult_Colon_bRP_Elite_50_f19.mgf",
        "Adult_Colon_bRP_Elite_50_f20.mgf",
        "Adult_Colon_bRP_Elite_50_f21.mgf",
        "Adult_Colon_bRP_Elite_50_f22.mgf",
        "Adult_Colon_bRP_Elite_50_f23.mgf",
        "Adult_Colon_bRP_Elite_50_f24.mgf"
    ]

    ADULT_ESOPHAGUS_BRP_VELOS_3_FILES = [
        "Adult_Esophagus_bRP_Velos_3_f01.mgf",
        "Adult_Esophagus_bRP_Velos_3_f02.mgf",
        "Adult_Esophagus_bRP_Velos_3_f03.mgf",
        "Adult_Esophagus_bRP_Velos_3_f04.mgf",
        "Adult_Esophagus_bRP_Velos_3_f05.mgf",
        "Adult_Esophagus_bRP_Velos_3_f06.mgf",
        "Adult_Esophagus_bRP_Velos_3_f07.mgf",
        "Adult_Esophagus_bRP_Velos_3_f08.mgf",
        "Adult_Esophagus_bRP_Velos_3_f09.mgf",
        "Adult_Esophagus_bRP_Velos_3_f10.mgf",
        "Adult_Esophagus_bRP_Velos_3_f11.mgf",
        "Adult_Esophagus_bRP_Velos_3_f12.mgf",
        "Adult_Esophagus_bRP_Velos_3_f13.mgf",
        "Adult_Esophagus_bRP_Velos_3_f14.mgf",
        "Adult_Esophagus_bRP_Velos_3_f15.mgf",
        "Adult_Esophagus_bRP_Velos_3_f16.mgf",
        "Adult_Esophagus_bRP_Velos_3_f17.mgf",
        "Adult_Esophagus_bRP_Velos_3_f18.mgf",
        "Adult_Esophagus_bRP_Velos_3_f19.mgf",
        "Adult_Esophagus_bRP_Velos_3_f20.mgf",
        "Adult_Esophagus_bRP_Velos_3_f21.mgf",
        "Adult_Esophagus_bRP_Velos_3_f22.mgf",
        "Adult_Esophagus_bRP_Velos_3_f23.mgf",
        "Adult_Esophagus_bRP_Velos_3_f24.mgf"
    ]

    ADULT_FRONTALCORTEX_GEL_ELITE_80_FILES = [
        "Adult_Frontalcortex_Gel_Elite_80_f01.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f02.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f03.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f04.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f05.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f06.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f07.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f08.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f09.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f10.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f11.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f12.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f13.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f14.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f15.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f16.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f17.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f18.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f19.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f20.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f21.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f22.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f23.mgf",
        "Adult_Frontalcortex_Gel_Elite_80_f24.mgf"
    ]

    ADULT_FRONTALCORTEX_BRP_ELITE_38_FILES = [
        "Adult_Frontalcortex_bRP_Elite_38_f01.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f02.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f03.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f04.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f05.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f06.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f07.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f08.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f09.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f10.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f11.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f12.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f13.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f14.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f15.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f16.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f17.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f18.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f19.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f20.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f21.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f22.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f23.mgf",
        "Adult_Frontalcortex_bRP_Elite_38_f24.mgf"
    ]

    ADULT_FRONTALCORTEX_BRP_ELITE_85_FILES = [
        "Adult_Frontalcortex_bRP_Elite_85_f01.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f02.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f03.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f04.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f05.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f06.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f07.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f08.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f09.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f10.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f11.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f12.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f13.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f14.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f15.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f16.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f17.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f18.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f19.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f20.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f21.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f22.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f23.mgf",
        "Adult_Frontalcortex_bRP_Elite_85_f24.mgf"
    ]

    ADULT_GALLBLADDER_GEL_ELITE_52_FILES = [
        "Adult_Gallbladder_Gel_Elite_52_f01.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f02.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f03.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f04.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f05.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f06.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f07.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f08.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f09.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f10.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f11.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f12.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f13.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f14.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f15.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f16.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f17.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f18.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f19.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f20.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f21.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f22.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f23.mgf",
        "Adult_Gallbladder_Gel_Elite_52_f24.mgf"
    ]

    ADULT_GALLBLADDER_BRP_ELITE_53_FILES = [
        "Adult_Gallbladder_bRP_Elite_53_f01.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f02.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f03.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f04.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f05.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f06.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f07.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f08.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f09.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f10.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f11.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f12.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f13.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f14.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f15.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f16.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f17.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f18.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f19.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f20.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f21.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f22.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f23.mgf",
        "Adult_Gallbladder_bRP_Elite_53_f24.mgf"
    ]

    ADULT_HEART_GEL_ELITE_54_FILES = [
        "Adult_Heart_Gel_Elite_54_f01.mgf",
        "Adult_Heart_Gel_Elite_54_f02.mgf",
        "Adult_Heart_Gel_Elite_54_f03.mgf",
        "Adult_Heart_Gel_Elite_54_f04.mgf",
        "Adult_Heart_Gel_Elite_54_f05.mgf",
        "Adult_Heart_Gel_Elite_54_f06.mgf",
        "Adult_Heart_Gel_Elite_54_f07.mgf",
        "Adult_Heart_Gel_Elite_54_f08.mgf",
        "Adult_Heart_Gel_Elite_54_f09.mgf",
        "Adult_Heart_Gel_Elite_54_f10.mgf",
        "Adult_Heart_Gel_Elite_54_f11.mgf",
        "Adult_Heart_Gel_Elite_54_f12.mgf",
        "Adult_Heart_Gel_Elite_54_f13.mgf",
        "Adult_Heart_Gel_Elite_54_f14.mgf",
        "Adult_Heart_Gel_Elite_54_f15.mgf",
        "Adult_Heart_Gel_Elite_54_f16.mgf",
        "Adult_Heart_Gel_Elite_54_f17.mgf",
        "Adult_Heart_Gel_Elite_54_f18.mgf",
        "Adult_Heart_Gel_Elite_54_f19.mgf",
        "Adult_Heart_Gel_Elite_54_f20.mgf",
        "Adult_Heart_Gel_Elite_54_f21.mgf",
        "Adult_Heart_Gel_Elite_54_f22.mgf",
        "Adult_Heart_Gel_Elite_54_f23.mgf",
        "Adult_Heart_Gel_Elite_54_f24.mgf"
    ]

    ADULT_HEART_GEL_VELOS_7_FILES = [
        "Adult_Heart_Gel_Velos_7_f01.mgf",
        "Adult_Heart_Gel_Velos_7_f02.mgf",
        "Adult_Heart_Gel_Velos_7_f03.mgf",
        "Adult_Heart_Gel_Velos_7_f04.mgf",
        "Adult_Heart_Gel_Velos_7_f05.mgf",
        "Adult_Heart_Gel_Velos_7_f06.mgf",
        "Adult_Heart_Gel_Velos_7_f07.mgf",
        "Adult_Heart_Gel_Velos_7_f08.mgf",
        "Adult_Heart_Gel_Velos_7_f09.mgf",
        "Adult_Heart_Gel_Velos_7_f10.mgf",
        "Adult_Heart_Gel_Velos_7_f11.mgf",
        "Adult_Heart_Gel_Velos_7_f12.mgf",
        "Adult_Heart_Gel_Velos_7_f13.mgf",
        "Adult_Heart_Gel_Velos_7_f14.mgf",
        "Adult_Heart_Gel_Velos_7_f15.mgf",
        "Adult_Heart_Gel_Velos_7_f16.mgf",
        "Adult_Heart_Gel_Velos_7_f17.mgf",
        "Adult_Heart_Gel_Velos_7_f18.mgf",
        "Adult_Heart_Gel_Velos_7_f19.mgf",
        "Adult_Heart_Gel_Velos_7_f20.mgf",
        "Adult_Heart_Gel_Velos_7_f21.mgf",
        "Adult_Heart_Gel_Velos_7_f22.mgf",
        "Adult_Heart_Gel_Velos_7_f23.mgf",
        "Adult_Heart_Gel_Velos_7_f24.mgf"
    ]

    ADULT_HEART_BRP_ELITE_81_FILES = [
        "Adult_Heart_bRP_Elite_81_f01.mgf",
        "Adult_Heart_bRP_Elite_81_f02.mgf",
        "Adult_Heart_bRP_Elite_81_f03.mgf",
        "Adult_Heart_bRP_Elite_81_f04.mgf",
        "Adult_Heart_bRP_Elite_81_f05.mgf",
        "Adult_Heart_bRP_Elite_81_f06.mgf",
        "Adult_Heart_bRP_Elite_81_f07.mgf",
        "Adult_Heart_bRP_Elite_81_f08.mgf",
        "Adult_Heart_bRP_Elite_81_f09.mgf",
        "Adult_Heart_bRP_Elite_81_f10.mgf",
        "Adult_Heart_bRP_Elite_81_f11.mgf",
        "Adult_Heart_bRP_Elite_81_f12.mgf",
        "Adult_Heart_bRP_Elite_81_f13.mgf",
        "Adult_Heart_bRP_Elite_81_f14.mgf",
        "Adult_Heart_bRP_Elite_81_f15.mgf",
        "Adult_Heart_bRP_Elite_81_f16.mgf",
        "Adult_Heart_bRP_Elite_81_f17.mgf",
        "Adult_Heart_bRP_Elite_81_f18.mgf",
        "Adult_Heart_bRP_Elite_81_f19.mgf",
        "Adult_Heart_bRP_Elite_81_f20.mgf",
        "Adult_Heart_bRP_Elite_81_f21.mgf",
        "Adult_Heart_bRP_Elite_81_f22.mgf",
        "Adult_Heart_bRP_Elite_81_f23.mgf",
        "Adult_Heart_bRP_Elite_81_f24.mgf"
    ]

    ADULT_KIDNEY_GEL_ELITE_55_FILES = [
        "Adult_Kidney_Gel_Elite_55_f01.mgf",
        "Adult_Kidney_Gel_Elite_55_f02.mgf",
        "Adult_Kidney_Gel_Elite_55_f03.mgf",
        "Adult_Kidney_Gel_Elite_55_f04.mgf",
        "Adult_Kidney_Gel_Elite_55_f05.mgf",
        "Adult_Kidney_Gel_Elite_55_f06.mgf",
        "Adult_Kidney_Gel_Elite_55_f07.mgf",
        "Adult_Kidney_Gel_Elite_55_f08.mgf",
        "Adult_Kidney_Gel_Elite_55_f09.mgf",
        "Adult_Kidney_Gel_Elite_55_f10.mgf",
        "Adult_Kidney_Gel_Elite_55_f11.mgf",
        "Adult_Kidney_Gel_Elite_55_f12.mgf",
        "Adult_Kidney_Gel_Elite_55_f13.mgf",
        "Adult_Kidney_Gel_Elite_55_f14.mgf",
        "Adult_Kidney_Gel_Elite_55_f15.mgf",
        "Adult_Kidney_Gel_Elite_55_f16.mgf",
        "Adult_Kidney_Gel_Elite_55_f17.mgf",
        "Adult_Kidney_Gel_Elite_55_f18.mgf",
        "Adult_Kidney_Gel_Elite_55_f19.mgf",
        "Adult_Kidney_Gel_Elite_55_f20.mgf",
        "Adult_Kidney_Gel_Elite_55_f21.mgf",
        "Adult_Kidney_Gel_Elite_55_f22.mgf",
        "Adult_Kidney_Gel_Elite_55_f23.mgf",
        "Adult_Kidney_Gel_Elite_55_f24.mgf",
        "Adult_Kidney_Gel_Elite_55_f25.mgf",
        "Adult_Kidney_Gel_Elite_55_f26.mgf",
        "Adult_Kidney_Gel_Elite_55_f27.mgf",
        "Adult_Kidney_Gel_Elite_55_f28.mgf",
        "Adult_Kidney_Gel_Elite_55_f29.mgf",
        "Adult_Kidney_Gel_Elite_55_f30.mgf",
        "Adult_Kidney_Gel_Elite_55_f31.mgf",
        "Adult_Kidney_Gel_Elite_55_f32.mgf"
    ]

    ADULT_KIDNEY_GEL_VELOS_9_FILES = [
        "Adult_Kidney_Gel_Velos_9_f01.mgf",
        "Adult_Kidney_Gel_Velos_9_f02.mgf",
        "Adult_Kidney_Gel_Velos_9_f03.mgf",
        "Adult_Kidney_Gel_Velos_9_f04.mgf",
        "Adult_Kidney_Gel_Velos_9_f05.mgf",
        "Adult_Kidney_Gel_Velos_9_f06.mgf",
        "Adult_Kidney_Gel_Velos_9_f07.mgf",
        "Adult_Kidney_Gel_Velos_9_f08.mgf",
        "Adult_Kidney_Gel_Velos_9_f09.mgf",
        "Adult_Kidney_Gel_Velos_9_f10.mgf",
        "Adult_Kidney_Gel_Velos_9_f11.mgf",
        "Adult_Kidney_Gel_Velos_9_f12.mgf",
        "Adult_Kidney_Gel_Velos_9_f13.mgf",
        "Adult_Kidney_Gel_Velos_9_f14.mgf",
        "Adult_Kidney_Gel_Velos_9_f15.mgf",
        "Adult_Kidney_Gel_Velos_9_f16.mgf",
        "Adult_Kidney_Gel_Velos_9_f17.mgf",
        "Adult_Kidney_Gel_Velos_9_f18.mgf",
        "Adult_Kidney_Gel_Velos_9_f19.mgf",
        "Adult_Kidney_Gel_Velos_9_f20.mgf",
        "Adult_Kidney_Gel_Velos_9_f21.mgf",
        "Adult_Kidney_Gel_Velos_9_f22.mgf",
        "Adult_Kidney_Gel_Velos_9_f23.mgf",
        "Adult_Kidney_Gel_Velos_9_f24.mgf",
        "Adult_Kidney_Gel_Velos_9_f25.mgf",
        "Adult_Kidney_Gel_Velos_9_f26.mgf",
        "Adult_Kidney_Gel_Velos_9_f27.mgf",
        "Adult_Kidney_Gel_Velos_9_f28.mgf",
        "Adult_Kidney_Gel_Velos_9_f29.mgf",
        "Adult_Kidney_Gel_Velos_9_f30.mgf",
        "Adult_Kidney_Gel_Velos_9_f31.mgf"
    ]

    ADULT_KIDNEY_BRP_VELOS_8_FILES = [
        "Adult_Kidney_bRP_Velos_8_f01.mgf",
        "Adult_Kidney_bRP_Velos_8_f02.mgf",
        "Adult_Kidney_bRP_Velos_8_f03.mgf",
        "Adult_Kidney_bRP_Velos_8_f04.mgf",
        "Adult_Kidney_bRP_Velos_8_f05.mgf",
        "Adult_Kidney_bRP_Velos_8_f06.mgf",
        "Adult_Kidney_bRP_Velos_8_f07.mgf",
        "Adult_Kidney_bRP_Velos_8_f08.mgf",
        "Adult_Kidney_bRP_Velos_8_f09.mgf",
        "Adult_Kidney_bRP_Velos_8_f10.mgf",
        "Adult_Kidney_bRP_Velos_8_f11.mgf",
        "Adult_Kidney_bRP_Velos_8_f12.mgf",
        "Adult_Kidney_bRP_Velos_8_f13.mgf",
        "Adult_Kidney_bRP_Velos_8_f14.mgf",
        "Adult_Kidney_bRP_Velos_8_f15.mgf",
        "Adult_Kidney_bRP_Velos_8_f16.mgf",
        "Adult_Kidney_bRP_Velos_8_f17.mgf",
        "Adult_Kidney_bRP_Velos_8_f18.mgf",
        "Adult_Kidney_bRP_Velos_8_f19.mgf",
        "Adult_Kidney_bRP_Velos_8_f20.mgf",
        "Adult_Kidney_bRP_Velos_8_f21.mgf",
        "Adult_Kidney_bRP_Velos_8_f22.mgf",
        "Adult_Kidney_bRP_Velos_8_f23.mgf",
        "Adult_Kidney_bRP_Velos_8_f24.mgf"
    ]

    ADULT_LIVER_GEL_ELILTE_83_FILES = [
        "Adult_Liver_Gel_Elite_83_f01.mgf",
        "Adult_Liver_Gel_Elite_83_f02.mgf",
        "Adult_Liver_Gel_Elite_83_f03.mgf",
        "Adult_Liver_Gel_Elite_83_f04.mgf",
        "Adult_Liver_Gel_Elite_83_f05.mgf",
        "Adult_Liver_Gel_Elite_83_f06.mgf",
        "Adult_Liver_Gel_Elite_83_f07.mgf",
        "Adult_Liver_Gel_Elite_83_f08.mgf",
        "Adult_Liver_Gel_Elite_83_f09.mgf",
        "Adult_Liver_Gel_Elite_83_f10.mgf",
        "Adult_Liver_Gel_Elite_83_f11.mgf",
        "Adult_Liver_Gel_Elite_83_f12.mgf",
        "Adult_Liver_Gel_Elite_83_f13.mgf",
        "Adult_Liver_Gel_Elite_83_f14.mgf",
        "Adult_Liver_Gel_Elite_83_f15.mgf",
        "Adult_Liver_Gel_Elite_83_f16.mgf",
        "Adult_Liver_Gel_Elite_83_f18.mgf",
        "Adult_Liver_Gel_Elite_83_f19.mgf",
        "Adult_Liver_Gel_Elite_83_f20.mgf",
        "Adult_Liver_Gel_Elite_83_f21.mgf",
        "Adult_Liver_Gel_Elite_83_f22.mgf",
        "Adult_Liver_Gel_Elite_83_f23.mgf",
        "Adult_Liver_Gel_Elite_83_f24.mgf"
    ]

    ADULT_LIVER_GEL_VELOS_11_FILES = [
        "Adult_Liver_Gel_Velos_11_f01.mgf",
        "Adult_Liver_Gel_Velos_11_f02.mgf",
        "Adult_Liver_Gel_Velos_11_f03.mgf",
        "Adult_Liver_Gel_Velos_11_f04.mgf",
        "Adult_Liver_Gel_Velos_11_f05.mgf",
        "Adult_Liver_Gel_Velos_11_f06.mgf",
        "Adult_Liver_Gel_Velos_11_f07.mgf",
        "Adult_Liver_Gel_Velos_11_f08.mgf",
        "Adult_Liver_Gel_Velos_11_f09.mgf",
        "Adult_Liver_Gel_Velos_11_f10.mgf",
        "Adult_Liver_Gel_Velos_11_f11.mgf",
        "Adult_Liver_Gel_Velos_11_f12.mgf",
        "Adult_Liver_Gel_Velos_11_f13.mgf",
        "Adult_Liver_Gel_Velos_11_f14.mgf",
        "Adult_Liver_Gel_Velos_11_f15.mgf",
        "Adult_Liver_Gel_Velos_11_f16.mgf",
        "Adult_Liver_Gel_Velos_11_f17.mgf",
        "Adult_Liver_Gel_Velos_11_f18.mgf",
        "Adult_Liver_Gel_Velos_11_f19.mgf",
        "Adult_Liver_Gel_Velos_11_f20.mgf",
        "Adult_Liver_Gel_Velos_11_f21.mgf",
        "Adult_Liver_Gel_Velos_11_f22.mgf",
        "Adult_Liver_Gel_Velos_11_f23.mgf",
        "Adult_Liver_Gel_Velos_11_f24.mgf",
        "Adult_Liver_Gel_Velos_11_f25.mgf",
        "Adult_Liver_Gel_Velos_11_f26.mgf"
    ]

    ADULT_LIVER_BRP_ELITE_82_FILES = [
        "Adult_Liver_bRP_Elite_82_f01.mgf",
        "Adult_Liver_bRP_Elite_82_f02.mgf",
        "Adult_Liver_bRP_Elite_82_f03.mgf",
        "Adult_Liver_bRP_Elite_82_f04.mgf",
        "Adult_Liver_bRP_Elite_82_f05.mgf",
        "Adult_Liver_bRP_Elite_82_f06.mgf",
        "Adult_Liver_bRP_Elite_82_f07.mgf",
        "Adult_Liver_bRP_Elite_82_f08.mgf",
        "Adult_Liver_bRP_Elite_82_f09.mgf",
        "Adult_Liver_bRP_Elite_82_f10.mgf",
        "Adult_Liver_bRP_Elite_82_f11.mgf",
        "Adult_Liver_bRP_Elite_82_f12.mgf",
        "Adult_Liver_bRP_Elite_82_f13.mgf",
        "Adult_Liver_bRP_Elite_82_f14.mgf",
        "Adult_Liver_bRP_Elite_82_f15.mgf",
        "Adult_Liver_bRP_Elite_82_f16.mgf",
        "Adult_Liver_bRP_Elite_82_f17.mgf",
        "Adult_Liver_bRP_Elite_82_f18.mgf",
        "Adult_Liver_bRP_Elite_82_f19.mgf",
        "Adult_Liver_bRP_Elite_82_f20.mgf",
        "Adult_Liver_bRP_Elite_82_f21.mgf",
        "Adult_Liver_bRP_Elite_82_f22.mgf",
        "Adult_Liver_bRP_Elite_82_f23.mgf",
        "Adult_Liver_bRP_Elite_82_f24.mgf"
    ]

    ADULT_LIVER_BRP_VELOS_10_FILES = [
        "Adult_Liver_bRP_Velos_10_f01.mgf",
        "Adult_Liver_bRP_Velos_10_f02.mgf",
        "Adult_Liver_bRP_Velos_10_f03.mgf",
        "Adult_Liver_bRP_Velos_10_f04.mgf",
        "Adult_Liver_bRP_Velos_10_f05.mgf",
        "Adult_Liver_bRP_Velos_10_f06.mgf",
        "Adult_Liver_bRP_Velos_10_f07.mgf",
        "Adult_Liver_bRP_Velos_10_f08.mgf",
        "Adult_Liver_bRP_Velos_10_f09.mgf",
        "Adult_Liver_bRP_Velos_10_f10.mgf",
        "Adult_Liver_bRP_Velos_10_f11.mgf",
        "Adult_Liver_bRP_Velos_10_f12.mgf",
        "Adult_Liver_bRP_Velos_10_f13.mgf",
        "Adult_Liver_bRP_Velos_10_f14.mgf",
        "Adult_Liver_bRP_Velos_10_f15.mgf",
        "Adult_Liver_bRP_Velos_10_f16.mgf",
        "Adult_Liver_bRP_Velos_10_f17.mgf",
        "Adult_Liver_bRP_Velos_10_f18.mgf",
        "Adult_Liver_bRP_Velos_10_f19.mgf",
        "Adult_Liver_bRP_Velos_10_f20.mgf",
        "Adult_Liver_bRP_Velos_10_f21.mgf",
        "Adult_Liver_bRP_Velos_10_f22.mgf",
        "Adult_Liver_bRP_Velos_10_f23.mgf",
        "Adult_Liver_bRP_Velos_10_f24.mgf",
        "Adult_Liver_bRP_Velos_10_f25.mgf"
    ]

    ADULT_LUNG_GEL_ELITE_56_FILES = [
        "Adult_Lung_Gel_Elite_56_f01.mgf",
        "Adult_Lung_Gel_Elite_56_f02.mgf",
        "Adult_Lung_Gel_Elite_56_f03.mgf",
        "Adult_Lung_Gel_Elite_56_f04.mgf",
        "Adult_Lung_Gel_Elite_56_f05.mgf",
        "Adult_Lung_Gel_Elite_56_f06.mgf",
        "Adult_Lung_Gel_Elite_56_f07.mgf",
        "Adult_Lung_Gel_Elite_56_f08.mgf",
        "Adult_Lung_Gel_Elite_56_f09.mgf",
        "Adult_Lung_Gel_Elite_56_f10.mgf",
        "Adult_Lung_Gel_Elite_56_f12.mgf",
        "Adult_Lung_Gel_Elite_56_f14.mgf",
        "Adult_Lung_Gel_Elite_56_f15.mgf",
        "Adult_Lung_Gel_Elite_56_f17.mgf",
        "Adult_Lung_Gel_Elite_56_f22.mgf",
        "Adult_Lung_Gel_Elite_56_f23.mgf"
    ]

    ADULT_LUNG_GEL_VELOS_13_FILES = [
        "Adult_Lung_Gel_Velos_13_f01.mgf",
        "Adult_Lung_Gel_Velos_13_f04.mgf",
        "Adult_Lung_Gel_Velos_13_f05.mgf",
        "Adult_Lung_Gel_Velos_13_f06.mgf",
        "Adult_Lung_Gel_Velos_13_f08.mgf",
        "Adult_Lung_Gel_Velos_13_f09.mgf",
        "Adult_Lung_Gel_Velos_13_f11.mgf",
        "Adult_Lung_Gel_Velos_13_f13.mgf",
        "Adult_Lung_Gel_Velos_13_f15.mgf",
        "Adult_Lung_Gel_Velos_13_f16.mgf",
        "Adult_Lung_Gel_Velos_13_f18.mgf",
        "Adult_Lung_Gel_Velos_13_f19.mgf",
        "Adult_Lung_Gel_Velos_13_f21.mgf",
        "Adult_Lung_Gel_Velos_13_f22.mgf",
        "Adult_Lung_Gel_Velos_13_f23.mgf",
        "Adult_Lung_Gel_Velos_13_f26.mgf"
    ]

    ADULT_LUNG_BRP_VELOS_12_FILES = [
        "Adult_Lung_bRP_Velos_12_f01.mgf",
        "Adult_Lung_bRP_Velos_12_f02.mgf",
        "Adult_Lung_bRP_Velos_12_f03.mgf",
        "Adult_Lung_bRP_Velos_12_f04.mgf",
        "Adult_Lung_bRP_Velos_12_f05.mgf",
        "Adult_Lung_bRP_Velos_12_f06.mgf",
        "Adult_Lung_bRP_Velos_12_f07.mgf",
        "Adult_Lung_bRP_Velos_12_f08.mgf",
        "Adult_Lung_bRP_Velos_12_f10.mgf",
        "Adult_Lung_bRP_Velos_12_f11.mgf",
        "Adult_Lung_bRP_Velos_12_f13.mgf",
        "Adult_Lung_bRP_Velos_12_f14.mgf",
        "Adult_Lung_bRP_Velos_12_f15.mgf",
        "Adult_Lung_bRP_Velos_12_f16.mgf",
        "Adult_Lung_bRP_Velos_12_f17.mgf",
        "Adult_Lung_bRP_Velos_12_f18.mgf",
        "Adult_Lung_bRP_Velos_12_f19.mgf",
        "Adult_Lung_bRP_Velos_12_f21.mgf",
        "Adult_Lung_bRP_Velos_12_f22.mgf",
        "Adult_Lung_bRP_Velos_12_f23.mgf",
        "Adult_Lung_bRP_Velos_12_f24.mgf"
    ]

    ADULT_MONOCYTES_GEL_VELOS_32_FILES = [
        "Adult_Monocytes_Gel_Velos_32_f01.mgf",
        "Adult_Monocytes_Gel_Velos_32_f02.mgf",
        "Adult_Monocytes_Gel_Velos_32_f03.mgf",
        "Adult_Monocytes_Gel_Velos_32_f05.mgf",
        "Adult_Monocytes_Gel_Velos_32_f06.mgf",
        "Adult_Monocytes_Gel_Velos_32_f07.mgf",
        "Adult_Monocytes_Gel_Velos_32_f08.mgf",
        "Adult_Monocytes_Gel_Velos_32_f09.mgf",
        "Adult_Monocytes_Gel_Velos_32_f10.mgf",
        "Adult_Monocytes_Gel_Velos_32_f13.mgf",
        "Adult_Monocytes_Gel_Velos_32_f14.mgf",
        "Adult_Monocytes_Gel_Velos_32_f15.mgf",
        "Adult_Monocytes_Gel_Velos_32_f18.mgf",
        "Adult_Monocytes_Gel_Velos_32_f19.mgf",
        "Adult_Monocytes_Gel_Velos_32_f20.mgf",
        "Adult_Monocytes_Gel_Velos_32_f21.mgf",
        "Adult_Monocytes_Gel_Velos_32_f23.mgf",
        "Adult_Monocytes_Gel_Velos_32_f24.mgf",
        "Adult_Monocytes_Gel_Velos_32_f25.mgf",
        "Adult_Monocytes_Gel_Velos_32_f26.mgf",
        "Adult_Monocytes_Gel_Velos_32_f27.mgf",
        "Adult_Monocytes_Gel_Velos_32_f28.mgf",
        "Adult_Monocytes_Gel_Velos_32_f29.mgf",
        #
        # Removing this file since Crux has not found any identifications here
        #
        # "Adult_Monocytes_Gel_Velos_32_f30.mgf",
        "Adult_Monocytes_Gel_Velos_32_f32.mgf",
        "Adult_Monocytes_Gel_Velos_32_f33.mgf",
        "Adult_Monocytes_Gel_Velos_32_f34.mgf",
        "Adult_Monocytes_Gel_Velos_32_f35.mgf",
        "Adult_Monocytes_Gel_Velos_32_f36.mgf",
        "Adult_Monocytes_Gel_Velos_32_f37.mgf",
        "Adult_Monocytes_Gel_Velos_32_f38.mgf",
        "Adult_Monocytes_Gel_Velos_32_f39.mgf"
    ]

    ADULT_MONOCYTES_BRP_ELITE_33_FILES = [
        "Adult_Monocytes_bRP_Elite_33_f01.mgf",
        "Adult_Monocytes_bRP_Elite_33_f02.mgf",
        "Adult_Monocytes_bRP_Elite_33_f03.mgf",
        "Adult_Monocytes_bRP_Elite_33_f04.mgf",
        "Adult_Monocytes_bRP_Elite_33_f05.mgf",
        "Adult_Monocytes_bRP_Elite_33_f06.mgf",
        "Adult_Monocytes_bRP_Elite_33_f07.mgf",
        "Adult_Monocytes_bRP_Elite_33_f08.mgf",
        "Adult_Monocytes_bRP_Elite_33_f09.mgf",
        "Adult_Monocytes_bRP_Elite_33_f10.mgf",
        "Adult_Monocytes_bRP_Elite_33_f11.mgf",
        "Adult_Monocytes_bRP_Elite_33_f12.mgf",
        "Adult_Monocytes_bRP_Elite_33_f13.mgf",
        "Adult_Monocytes_bRP_Elite_33_f14.mgf",
        "Adult_Monocytes_bRP_Elite_33_f16.mgf",
        "Adult_Monocytes_bRP_Elite_33_f17.mgf",
        "Adult_Monocytes_bRP_Elite_33_f18.mgf",
        "Adult_Monocytes_bRP_Elite_33_f19.mgf",
        "Adult_Monocytes_bRP_Elite_33_f20.mgf",
        "Adult_Monocytes_bRP_Elite_33_f21.mgf",
        "Adult_Monocytes_bRP_Elite_33_f23.mgf",
        "Adult_Monocytes_bRP_Elite_33_f26.mgf"
    ]

    ADULT_MONOCYTES_BRP_VELOS_31_FILES = [
        "Adult_Monocytes_bRP_Velos_31_f03.mgf",
        "Adult_Monocytes_bRP_Velos_31_f04.mgf",
        "Adult_Monocytes_bRP_Velos_31_f06.mgf",
        "Adult_Monocytes_bRP_Velos_31_f08.mgf",
        "Adult_Monocytes_bRP_Velos_31_f09.mgf",
        "Adult_Monocytes_bRP_Velos_31_f10.mgf",
        "Adult_Monocytes_bRP_Velos_31_f11.mgf",
        "Adult_Monocytes_bRP_Velos_31_f12.mgf",
        "Adult_Monocytes_bRP_Velos_31_f13.mgf",
        "Adult_Monocytes_bRP_Velos_31_f14.mgf",
        "Adult_Monocytes_bRP_Velos_31_f15.mgf",
        "Adult_Monocytes_bRP_Velos_31_f17.mgf",
        "Adult_Monocytes_bRP_Velos_31_f18.mgf",
        "Adult_Monocytes_bRP_Velos_31_f19.mgf",
        "Adult_Monocytes_bRP_Velos_31_f20.mgf",
        "Adult_Monocytes_bRP_Velos_31_f21.mgf",
        "Adult_Monocytes_bRP_Velos_31_f22.mgf",
        "Adult_Monocytes_bRP_Velos_31_f23.mgf",
        "Adult_Monocytes_bRP_Velos_31_f24.mgf",
        "Adult_Monocytes_bRP_Velos_31_f25.mgf"
    ]

    ADULT_NKCELLS_GEL_ELITE_78_FILES = [
        "Adult_NKcells_Gel_Elite_78_f01.mgf",
        "Adult_NKcells_Gel_Elite_78_f02.mgf",
        "Adult_NKcells_Gel_Elite_78_f03.mgf",
        "Adult_NKcells_Gel_Elite_78_f04.mgf",
        "Adult_NKcells_Gel_Elite_78_f05.mgf",
        "Adult_NKcells_Gel_Elite_78_f06.mgf",
        "Adult_NKcells_Gel_Elite_78_f07.mgf",
        "Adult_NKcells_Gel_Elite_78_f08.mgf",
        "Adult_NKcells_Gel_Elite_78_f09.mgf",
        "Adult_NKcells_Gel_Elite_78_f10.mgf",
        "Adult_NKcells_Gel_Elite_78_f11.mgf",
        "Adult_NKcells_Gel_Elite_78_f12.mgf",
        "Adult_NKcells_Gel_Elite_78_f13.mgf",
        "Adult_NKcells_Gel_Elite_78_f14.mgf",
        "Adult_NKcells_Gel_Elite_78_f15.mgf",
        "Adult_NKcells_Gel_Elite_78_f16.mgf",
        "Adult_NKcells_Gel_Elite_78_f17.mgf",
        "Adult_NKcells_Gel_Elite_78_f18.mgf",
        "Adult_NKcells_Gel_Elite_78_f19.mgf",
        "Adult_NKcells_Gel_Elite_78_f20.mgf",
        "Adult_NKcells_Gel_Elite_78_f21.mgf",
        "Adult_NKcells_Gel_Elite_78_f22.mgf",
        "Adult_NKcells_Gel_Elite_78_f23.mgf",
        "Adult_NKcells_Gel_Elite_78_f24.mgf"
    ]

    ADULT_NKCELLS_GEL_VELOS_47_FILES = [
        "Adult_NKcells_Gel_Velos_47_f01.mgf",
        "Adult_NKcells_Gel_Velos_47_f02.mgf",
        "Adult_NKcells_Gel_Velos_47_f03.mgf",
        "Adult_NKcells_Gel_Velos_47_f05.mgf",
        "Adult_NKcells_Gel_Velos_47_f06.mgf",
        "Adult_NKcells_Gel_Velos_47_f07.mgf",
        "Adult_NKcells_Gel_Velos_47_f08.mgf",
        "Adult_NKcells_Gel_Velos_47_f09.mgf",
        "Adult_NKcells_Gel_Velos_47_f10.mgf",
        "Adult_NKcells_Gel_Velos_47_f11.mgf",
        "Adult_NKcells_Gel_Velos_47_f12.mgf",
        "Adult_NKcells_Gel_Velos_47_f13.mgf",
        "Adult_NKcells_Gel_Velos_47_f14.mgf",
        "Adult_NKcells_Gel_Velos_47_f15.mgf",
        "Adult_NKcells_Gel_Velos_47_f16.mgf",
        "Adult_NKcells_Gel_Velos_47_f17.mgf",
        "Adult_NKcells_Gel_Velos_47_f18.mgf",
        "Adult_NKcells_Gel_Velos_47_f19.mgf",
        "Adult_NKcells_Gel_Velos_47_f20.mgf",
        "Adult_NKcells_Gel_Velos_47_f21.mgf",
        "Adult_NKcells_Gel_Velos_47_f22.mgf",
        "Adult_NKcells_Gel_Velos_47_f23.mgf",
        "Adult_NKcells_Gel_Velos_47_f24.mgf",
        "Adult_NKcells_Gel_Velos_47_f25.mgf",
        "Adult_NKcells_Gel_Velos_47_f26.mgf",
        "Adult_NKcells_Gel_Velos_47_f27.mgf",
        "Adult_NKcells_Gel_Velos_47_f28.mgf",
        "Adult_NKcells_Gel_Velos_47_f29.mgf",
        "Adult_NKcells_Gel_Velos_47_f30.mgf"
    ]

    ADULT_OVARY_GEL_ELITE_58_FILES = [
        "Adult_Ovary_Gel_Elite_58_f01.mgf",
        "Adult_Ovary_Gel_Elite_58_f02.mgf",
        "Adult_Ovary_Gel_Elite_58_f03.mgf",
        "Adult_Ovary_Gel_Elite_58_f04.mgf",
        "Adult_Ovary_Gel_Elite_58_f05.mgf",
        "Adult_Ovary_Gel_Elite_58_f06.mgf",
        "Adult_Ovary_Gel_Elite_58_f07.mgf",
        "Adult_Ovary_Gel_Elite_58_f08.mgf",
        "Adult_Ovary_Gel_Elite_58_f09.mgf",
        "Adult_Ovary_Gel_Elite_58_f10.mgf",
        "Adult_Ovary_Gel_Elite_58_f11.mgf",
        "Adult_Ovary_Gel_Elite_58_f12.mgf",
        "Adult_Ovary_Gel_Elite_58_f13.mgf",
        "Adult_Ovary_Gel_Elite_58_f14.mgf",
        "Adult_Ovary_Gel_Elite_58_f15.mgf",
        "Adult_Ovary_Gel_Elite_58_f16.mgf",
        "Adult_Ovary_Gel_Elite_58_f17.mgf",
        "Adult_Ovary_Gel_Elite_58_f18.mgf",
        "Adult_Ovary_Gel_Elite_58_f19.mgf",
        "Adult_Ovary_Gel_Elite_58_f20.mgf",
        "Adult_Ovary_Gel_Elite_58_f21.mgf",
        "Adult_Ovary_Gel_Elite_58_f22.mgf",
        "Adult_Ovary_Gel_Elite_58_f23.mgf",
        "Adult_Ovary_Gel_Elite_58_f24.mgf"
    ]

    ADULT_OVARY_BRP_ELITE_57_FILES = [
        "Adult_Ovary_bRP_Elite_57_f01.mgf",
        "Adult_Ovary_bRP_Elite_57_f02.mgf",
        "Adult_Ovary_bRP_Elite_57_f03.mgf",
        "Adult_Ovary_bRP_Elite_57_f04.mgf",
        "Adult_Ovary_bRP_Elite_57_f05.mgf",
        "Adult_Ovary_bRP_Elite_57_f06.mgf",
        "Adult_Ovary_bRP_Elite_57_f07.mgf",
        "Adult_Ovary_bRP_Elite_57_f08.mgf",
        "Adult_Ovary_bRP_Elite_57_f09.mgf",
        "Adult_Ovary_bRP_Elite_57_f10.mgf",
        "Adult_Ovary_bRP_Elite_57_f11.mgf",
        "Adult_Ovary_bRP_Elite_57_f12.mgf",
        "Adult_Ovary_bRP_Elite_57_f13.mgf",
        "Adult_Ovary_bRP_Elite_57_f14.mgf",
        "Adult_Ovary_bRP_Elite_57_f15.mgf",
        "Adult_Ovary_bRP_Elite_57_f16.mgf",
        "Adult_Ovary_bRP_Elite_57_f17.mgf",
        "Adult_Ovary_bRP_Elite_57_f18.mgf",
        "Adult_Ovary_bRP_Elite_57_f19.mgf",
        "Adult_Ovary_bRP_Elite_57_f20.mgf",
        "Adult_Ovary_bRP_Elite_57_f21.mgf",
        "Adult_Ovary_bRP_Elite_57_f22.mgf",
        "Adult_Ovary_bRP_Elite_57_f23.mgf",
        "Adult_Ovary_bRP_Elite_57_f24.mgf"
    ]

    ADULT_PANCREAS_GEL_ELITE_60_FILES = [
        "Adult_Pancreas_Gel_Elite_60_f01.mgf",
        "Adult_Pancreas_Gel_Elite_60_f02.mgf",
        "Adult_Pancreas_Gel_Elite_60_f03.mgf",
        "Adult_Pancreas_Gel_Elite_60_f04.mgf",
        "Adult_Pancreas_Gel_Elite_60_f05.mgf",
        "Adult_Pancreas_Gel_Elite_60_f06.mgf",
        "Adult_Pancreas_Gel_Elite_60_f07.mgf",
        "Adult_Pancreas_Gel_Elite_60_f08.mgf",
        "Adult_Pancreas_Gel_Elite_60_f09.mgf",
        "Adult_Pancreas_Gel_Elite_60_f10.mgf",
        "Adult_Pancreas_Gel_Elite_60_f11.mgf",
        "Adult_Pancreas_Gel_Elite_60_f12.mgf",
        "Adult_Pancreas_Gel_Elite_60_f13.mgf",
        "Adult_Pancreas_Gel_Elite_60_f14.mgf",
        "Adult_Pancreas_Gel_Elite_60_f15.mgf",
        "Adult_Pancreas_Gel_Elite_60_f16.mgf",
        "Adult_Pancreas_Gel_Elite_60_f17.mgf",
        "Adult_Pancreas_Gel_Elite_60_f18.mgf",
        "Adult_Pancreas_Gel_Elite_60_f19.mgf",
        "Adult_Pancreas_Gel_Elite_60_f20.mgf",
        "Adult_Pancreas_Gel_Elite_60_f21.mgf",
        "Adult_Pancreas_Gel_Elite_60_f22.mgf",
        "Adult_Pancreas_Gel_Elite_60_f23.mgf",
        "Adult_Pancreas_Gel_Elite_60_f24.mgf"
    ]

    ADULT_PLATELETS_GEL_VELOS_36_FILES = [
        "Adult_Platelets_Gel_Velos_36_f01.mgf",
        "Adult_Platelets_Gel_Velos_36_f02.mgf",
        "Adult_Platelets_Gel_Velos_36_f03.mgf",
        "Adult_Platelets_Gel_Velos_36_f04.mgf",
        "Adult_Platelets_Gel_Velos_36_f05.mgf",
        "Adult_Platelets_Gel_Velos_36_f06.mgf",
        "Adult_Platelets_Gel_Velos_36_f07.mgf",
        "Adult_Platelets_Gel_Velos_36_f08.mgf",
        "Adult_Platelets_Gel_Velos_36_f09.mgf",
        "Adult_Platelets_Gel_Velos_36_f10.mgf",
        "Adult_Platelets_Gel_Velos_36_f11.mgf",
        "Adult_Platelets_Gel_Velos_36_f12.mgf",
        "Adult_Platelets_Gel_Velos_36_f13.mgf",
        "Adult_Platelets_Gel_Velos_36_f14.mgf",
        "Adult_Platelets_Gel_Velos_36_f15.mgf",
        "Adult_Platelets_Gel_Velos_36_f16.mgf",
        "Adult_Platelets_Gel_Velos_36_f17.mgf",
        "Adult_Platelets_Gel_Velos_36_f18.mgf",
        "Adult_Platelets_Gel_Velos_36_f19.mgf",
        "Adult_Platelets_Gel_Velos_36_f20.mgf",
        "Adult_Platelets_Gel_Velos_36_f21.mgf",
        "Adult_Platelets_Gel_Velos_36_f22.mgf",
        "Adult_Platelets_Gel_Velos_36_f23.mgf",
        "Adult_Platelets_Gel_Velos_36_f24.mgf",
        "Adult_Platelets_Gel_Velos_36_f25.mgf",
        "Adult_Platelets_Gel_Velos_36_f26.mgf",
        "Adult_Platelets_Gel_Velos_36_f27.mgf",
        "Adult_Platelets_Gel_Velos_36_f28.mgf",
        "Adult_Platelets_Gel_Velos_36_f29.mgf",
        "Adult_Platelets_Gel_Velos_36_f30.mgf",
        "Adult_Platelets_Gel_Velos_36_f31.mgf",
        "Adult_Platelets_Gel_Velos_36_f32.mgf",
        "Adult_Platelets_Gel_Velos_36_f33.mgf",
        "Adult_Platelets_Gel_Velos_36_f34.mgf",
        "Adult_Platelets_Gel_Velos_36_f35.mgf",
        "Adult_Platelets_Gel_Velos_36_f36.mgf",
        "Adult_Platelets_Gel_Velos_36_f37.mgf",
        "Adult_Platelets_Gel_Velos_36_f38.mgf"
    ]

    ADULT_PLATELETS_BRP_VELOS_35_FILES = [
        "Adult_Platelets_bRP_Velos_35_f01.mgf",
        "Adult_Platelets_bRP_Velos_35_f02.mgf",
        "Adult_Platelets_bRP_Velos_35_f03.mgf",
        "Adult_Platelets_bRP_Velos_35_f04.mgf",
        "Adult_Platelets_bRP_Velos_35_f05.mgf",
        "Adult_Platelets_bRP_Velos_35_f06.mgf",
        "Adult_Platelets_bRP_Velos_35_f07.mgf",
        "Adult_Platelets_bRP_Velos_35_f08.mgf",
        "Adult_Platelets_bRP_Velos_35_f09.mgf",
        "Adult_Platelets_bRP_Velos_35_f10.mgf",
        "Adult_Platelets_bRP_Velos_35_f11.mgf",
        "Adult_Platelets_bRP_Velos_35_f12.mgf",
        "Adult_Platelets_bRP_Velos_35_f13.mgf",
        "Adult_Platelets_bRP_Velos_35_f14.mgf",
        "Adult_Platelets_bRP_Velos_35_f15.mgf",
        "Adult_Platelets_bRP_Velos_35_f16.mgf",
        "Adult_Platelets_bRP_Velos_35_f17.mgf",
        "Adult_Platelets_bRP_Velos_35_f18.mgf",
        "Adult_Platelets_bRP_Velos_35_f19.mgf",
        "Adult_Platelets_bRP_Velos_35_f20.mgf",
        "Adult_Platelets_bRP_Velos_35_f21.mgf",
        "Adult_Platelets_bRP_Velos_35_f22.mgf",
        "Adult_Platelets_bRP_Velos_35_f23.mgf",
        "Adult_Platelets_bRP_Velos_35_f24.mgf",
        "Adult_Platelets_bRP_Velos_35_f25.mgf"
    ]

    ADULT_PROSTATE_GEL_ELITE_62_FILES = [
        "Adult_Prostate_Gel_Elite_62_f01.mgf",
        "Adult_Prostate_Gel_Elite_62_f02.mgf",
        "Adult_Prostate_Gel_Elite_62_f03.mgf",
        "Adult_Prostate_Gel_Elite_62_f04.mgf",
        "Adult_Prostate_Gel_Elite_62_f05.mgf",
        "Adult_Prostate_Gel_Elite_62_f06.mgf",
        "Adult_Prostate_Gel_Elite_62_f07.mgf",
        "Adult_Prostate_Gel_Elite_62_f08.mgf",
        "Adult_Prostate_Gel_Elite_62_f09.mgf",
        "Adult_Prostate_Gel_Elite_62_f10.mgf",
        "Adult_Prostate_Gel_Elite_62_f11.mgf",
        "Adult_Prostate_Gel_Elite_62_f12.mgf",
        "Adult_Prostate_Gel_Elite_62_f13.mgf",
        "Adult_Prostate_Gel_Elite_62_f14.mgf",
        "Adult_Prostate_Gel_Elite_62_f15.mgf",
        "Adult_Prostate_Gel_Elite_62_f16.mgf",
        "Adult_Prostate_Gel_Elite_62_f17.mgf",
        "Adult_Prostate_Gel_Elite_62_f18.mgf",
        "Adult_Prostate_Gel_Elite_62_f19.mgf",
        "Adult_Prostate_Gel_Elite_62_f20.mgf",
        "Adult_Prostate_Gel_Elite_62_f21.mgf",
        "Adult_Prostate_Gel_Elite_62_f22.mgf",
        "Adult_Prostate_Gel_Elite_62_f23.mgf",
        "Adult_Prostate_Gel_Elite_62_f24.mgf"
    ]

    ADULT_PROSTATE_BRP_ELITE_61_FILES = [
        "Adult_Prostate_bRP_Elite_61_f01.mgf",
        "Adult_Prostate_bRP_Elite_61_f02.mgf",
        "Adult_Prostate_bRP_Elite_61_f03.mgf",
        "Adult_Prostate_bRP_Elite_61_f04.mgf",
        "Adult_Prostate_bRP_Elite_61_f05.mgf",
        "Adult_Prostate_bRP_Elite_61_f06.mgf",
        "Adult_Prostate_bRP_Elite_61_f07.mgf",
        "Adult_Prostate_bRP_Elite_61_f08.mgf",
        "Adult_Prostate_bRP_Elite_61_f09.mgf",
        "Adult_Prostate_bRP_Elite_61_f10.mgf",
        "Adult_Prostate_bRP_Elite_61_f11.mgf",
        "Adult_Prostate_bRP_Elite_61_f12.mgf",
        "Adult_Prostate_bRP_Elite_61_f13.mgf",
        "Adult_Prostate_bRP_Elite_61_f14.mgf",
        "Adult_Prostate_bRP_Elite_61_f15.mgf",
        "Adult_Prostate_bRP_Elite_61_f16.mgf",
        "Adult_Prostate_bRP_Elite_61_f17.mgf",
        "Adult_Prostate_bRP_Elite_61_f18.mgf",
        "Adult_Prostate_bRP_Elite_61_f19.mgf",
        "Adult_Prostate_bRP_Elite_61_f20.mgf",
        "Adult_Prostate_bRP_Elite_61_f21.mgf",
        "Adult_Prostate_bRP_Elite_61_f22.mgf",
        "Adult_Prostate_bRP_Elite_61_f23.mgf",
        "Adult_Prostate_bRP_Elite_61_f24.mgf"
    ]

    ADULT_RECTUM_GEL_ELITE_63_FILES = [
        "Adult_Rectum_Gel_Elite_63_f01.mgf",
        "Adult_Rectum_Gel_Elite_63_f02.mgf",
        "Adult_Rectum_Gel_Elite_63_f03.mgf",
        "Adult_Rectum_Gel_Elite_63_f04.mgf",
        "Adult_Rectum_Gel_Elite_63_f05.mgf",
        "Adult_Rectum_Gel_Elite_63_f06.mgf",
        "Adult_Rectum_Gel_Elite_63_f07.mgf",
        "Adult_Rectum_Gel_Elite_63_f08.mgf",
        "Adult_Rectum_Gel_Elite_63_f09.mgf",
        "Adult_Rectum_Gel_Elite_63_f10.mgf",
        "Adult_Rectum_Gel_Elite_63_f11.mgf",
        "Adult_Rectum_Gel_Elite_63_f12.mgf",
        "Adult_Rectum_Gel_Elite_63_f13.mgf",
        "Adult_Rectum_Gel_Elite_63_f14.mgf",
        "Adult_Rectum_Gel_Elite_63_f15.mgf",
        "Adult_Rectum_Gel_Elite_63_f16.mgf",
        "Adult_Rectum_Gel_Elite_63_f17.mgf",
        "Adult_Rectum_Gel_Elite_63_f18.mgf",
        "Adult_Rectum_Gel_Elite_63_f19.mgf",
        "Adult_Rectum_Gel_Elite_63_f20.mgf",
        "Adult_Rectum_Gel_Elite_63_f21.mgf",
        "Adult_Rectum_Gel_Elite_63_f22.mgf",
        "Adult_Rectum_Gel_Elite_63_f23.mgf",
        "Adult_Rectum_Gel_Elite_63_f24.mgf",
        "Adult_Rectum_Gel_Elite_63_f25.mgf"
    ]

    ADULT_RECTUM_BRP_ELITE_84_FILES = [
        "Adult_Rectum_bRP_Elite_84_f01.mgf",
        "Adult_Rectum_bRP_Elite_84_f02.mgf",
        "Adult_Rectum_bRP_Elite_84_f03.mgf",
        "Adult_Rectum_bRP_Elite_84_f04.mgf",
        "Adult_Rectum_bRP_Elite_84_f05.mgf",
        "Adult_Rectum_bRP_Elite_84_f06.mgf",
        "Adult_Rectum_bRP_Elite_84_f07.mgf",
        "Adult_Rectum_bRP_Elite_84_f08.mgf",
        "Adult_Rectum_bRP_Elite_84_f09.mgf",
        "Adult_Rectum_bRP_Elite_84_f10.mgf",
        "Adult_Rectum_bRP_Elite_84_f11.mgf",
        "Adult_Rectum_bRP_Elite_84_f12.mgf",
        "Adult_Rectum_bRP_Elite_84_f13.mgf",
        "Adult_Rectum_bRP_Elite_84_f14.mgf",
        "Adult_Rectum_bRP_Elite_84_f15.mgf",
        "Adult_Rectum_bRP_Elite_84_f16.mgf",
        "Adult_Rectum_bRP_Elite_84_f17.mgf",
        "Adult_Rectum_bRP_Elite_84_f18.mgf",
        "Adult_Rectum_bRP_Elite_84_f19.mgf",
        "Adult_Rectum_bRP_Elite_84_f20.mgf",
        "Adult_Rectum_bRP_Elite_84_f21.mgf",
        "Adult_Rectum_bRP_Elite_84_f22.mgf",
        "Adult_Rectum_bRP_Elite_84_f23.mgf",
        "Adult_Rectum_bRP_Elite_84_f24.mgf"
    ]

    ADULT_RETINA_GEL_ELITE_65_FILES = [
        "Adult_Retina_Gel_Elite_65_f01.mgf",
        "Adult_Retina_Gel_Elite_65_f02.mgf",
        "Adult_Retina_Gel_Elite_65_f03.mgf",
        "Adult_Retina_Gel_Elite_65_f04.mgf",
        "Adult_Retina_Gel_Elite_65_f05.mgf",
        "Adult_Retina_Gel_Elite_65_f06.mgf",
        "Adult_Retina_Gel_Elite_65_f07.mgf",
        "Adult_Retina_Gel_Elite_65_f08.mgf",
        "Adult_Retina_Gel_Elite_65_f09.mgf",
        "Adult_Retina_Gel_Elite_65_f10.mgf",
        "Adult_Retina_Gel_Elite_65_f11.mgf",
        "Adult_Retina_Gel_Elite_65_f12.mgf",
        "Adult_Retina_Gel_Elite_65_f13.mgf",
        "Adult_Retina_Gel_Elite_65_f14.mgf",
        "Adult_Retina_Gel_Elite_65_f15.mgf",
        "Adult_Retina_Gel_Elite_65_f16.mgf",
        "Adult_Retina_Gel_Elite_65_f17.mgf",
        "Adult_Retina_Gel_Elite_65_f18.mgf",
        "Adult_Retina_Gel_Elite_65_f19.mgf",
        "Adult_Retina_Gel_Elite_65_f20.mgf",
        "Adult_Retina_Gel_Elite_65_f21.mgf",
        "Adult_Retina_Gel_Elite_65_f22.mgf",
        "Adult_Retina_Gel_Elite_65_f23.mgf",
        "Adult_Retina_Gel_Elite_65_f24.mgf"
    ]

    ADULT_RETINA_GEL_VELOS_5_FILES = [
        "Adult_Retina_Gel_Velos_5_f01.mgf",
        "Adult_Retina_Gel_Velos_5_f02.mgf",
        "Adult_Retina_Gel_Velos_5_f03.mgf",
        "Adult_Retina_Gel_Velos_5_f04.mgf",
        "Adult_Retina_Gel_Velos_5_f05.mgf",
        "Adult_Retina_Gel_Velos_5_f06.mgf",
        "Adult_Retina_Gel_Velos_5_f07.mgf",
        "Adult_Retina_Gel_Velos_5_f08.mgf",
        "Adult_Retina_Gel_Velos_5_f09.mgf",
        "Adult_Retina_Gel_Velos_5_f10.mgf",
        "Adult_Retina_Gel_Velos_5_f11.mgf",
        "Adult_Retina_Gel_Velos_5_f12.mgf",
        "Adult_Retina_Gel_Velos_5_f13.mgf",
        "Adult_Retina_Gel_Velos_5_f14.mgf",
        "Adult_Retina_Gel_Velos_5_f15.mgf",
        "Adult_Retina_Gel_Velos_5_f16.mgf",
        "Adult_Retina_Gel_Velos_5_f17.mgf",
        "Adult_Retina_Gel_Velos_5_f18.mgf",
        "Adult_Retina_Gel_Velos_5_f19.mgf",
        "Adult_Retina_Gel_Velos_5_f20.mgf",
        "Adult_Retina_Gel_Velos_5_f21.mgf",
        "Adult_Retina_Gel_Velos_5_f22.mgf",
        "Adult_Retina_Gel_Velos_5_f23.mgf",
        "Adult_Retina_Gel_Velos_5_f24.mgf",
        "Adult_Retina_Gel_Velos_5_f25.mgf",
        "Adult_Retina_Gel_Velos_5_f26.mgf",
        "Adult_Retina_Gel_Velos_5_f27.mgf",
        "Adult_Retina_Gel_Velos_5_f28.mgf",
        "Adult_Retina_Gel_Velos_5_f29.mgf",
        "Adult_Retina_Gel_Velos_5_f30.mgf",
        "Adult_Retina_Gel_Velos_5_f31.mgf"
    ]

    ADULT_RETINA_BRP_ELITE_64_FILES = [
        "Adult_Retina_bRP_Elite_64_f01.mgf",
        "Adult_Retina_bRP_Elite_64_f02.mgf",
        "Adult_Retina_bRP_Elite_64_f03.mgf",
        "Adult_Retina_bRP_Elite_64_f04.mgf",
        "Adult_Retina_bRP_Elite_64_f05.mgf",
        "Adult_Retina_bRP_Elite_64_f06.mgf",
        "Adult_Retina_bRP_Elite_64_f07.mgf",
        "Adult_Retina_bRP_Elite_64_f08.mgf",
        "Adult_Retina_bRP_Elite_64_f09.mgf",
        "Adult_Retina_bRP_Elite_64_f10.mgf",
        "Adult_Retina_bRP_Elite_64_f11.mgf",
        "Adult_Retina_bRP_Elite_64_f12.mgf",
        "Adult_Retina_bRP_Elite_64_f13.mgf",
        "Adult_Retina_bRP_Elite_64_f14.mgf",
        "Adult_Retina_bRP_Elite_64_f15.mgf",
        "Adult_Retina_bRP_Elite_64_f16.mgf",
        "Adult_Retina_bRP_Elite_64_f17.mgf",
        "Adult_Retina_bRP_Elite_64_f18.mgf",
        "Adult_Retina_bRP_Elite_64_f19.mgf",
        "Adult_Retina_bRP_Elite_64_f20.mgf",
        "Adult_Retina_bRP_Elite_64_f21.mgf",
        "Adult_Retina_bRP_Elite_64_f22.mgf",
        "Adult_Retina_bRP_Elite_64_f23.mgf",
        "Adult_Retina_bRP_Elite_64_f24.mgf"
    ]


    ADULT_SPINALCORD_GEL_ELITE_67_FILES = [
        "Adult_Spinalcord_Gel_Elite_67_f01.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f02.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f03.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f04.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f05.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f06.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f07.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f08.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f09.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f10.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f11.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f12.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f13.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f14.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f15.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f16.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f17.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f18.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f19.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f20.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f21.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f22.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f23.mgf",
        "Adult_Spinalcord_Gel_Elite_67_f24.mgf"
    ]


    ADULT_SPINALCORD_BRP_ELITE_66_FILES = [
        "Adult_Spinalcord_bRP_Elite_66_f01.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f02.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f03.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f04.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f05.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f06.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f07.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f08.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f09.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f10.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f11.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f12.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f13.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f14.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f15.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f16.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f17.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f18.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f19.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f20.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f21.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f22.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f23.mgf",
        "Adult_Spinalcord_bRP_Elite_66_f24.mgf"
    ]


    ADULT_TESTIS_GEL_ELITE_69_FILES = [
        "Adult_Testis_Gel_Elite_69_f01.mgf",
        "Adult_Testis_Gel_Elite_69_f02.mgf",
        "Adult_Testis_Gel_Elite_69_f03.mgf",
        "Adult_Testis_Gel_Elite_69_f04.mgf",
        "Adult_Testis_Gel_Elite_69_f05.mgf",
        "Adult_Testis_Gel_Elite_69_f06.mgf",
        "Adult_Testis_Gel_Elite_69_f07.mgf",
        "Adult_Testis_Gel_Elite_69_f08.mgf",
        "Adult_Testis_Gel_Elite_69_f09.mgf",
        "Adult_Testis_Gel_Elite_69_f10.mgf",
        "Adult_Testis_Gel_Elite_69_f11.mgf",
        "Adult_Testis_Gel_Elite_69_f12.mgf",
        "Adult_Testis_Gel_Elite_69_f13.mgf",
        "Adult_Testis_Gel_Elite_69_f14.mgf",
        "Adult_Testis_Gel_Elite_69_f15.mgf",
        "Adult_Testis_Gel_Elite_69_f16.mgf",
        "Adult_Testis_Gel_Elite_69_f17.mgf",
        "Adult_Testis_Gel_Elite_69_f18.mgf",
        "Adult_Testis_Gel_Elite_69_f19.mgf",
        "Adult_Testis_Gel_Elite_69_f20.mgf",
        "Adult_Testis_Gel_Elite_69_f21.mgf",
        "Adult_Testis_Gel_Elite_69_f22.mgf",
        "Adult_Testis_Gel_Elite_69_f23.mgf",
        "Adult_Testis_Gel_Elite_69_f24.mgf"
    ]


    ADULT_TESTIS_BRP_ELITE_68_FILES = [
        "Adult_Testis_bRP_Elite_68_f01.mgf",
        "Adult_Testis_bRP_Elite_68_f02.mgf",
        "Adult_Testis_bRP_Elite_68_f03.mgf",
        "Adult_Testis_bRP_Elite_68_f04.mgf",
        "Adult_Testis_bRP_Elite_68_f05.mgf",
        "Adult_Testis_bRP_Elite_68_f06.mgf",
        "Adult_Testis_bRP_Elite_68_f07.mgf",
        "Adult_Testis_bRP_Elite_68_f08.mgf",
        "Adult_Testis_bRP_Elite_68_f09.mgf",
        "Adult_Testis_bRP_Elite_68_f10.mgf",
        "Adult_Testis_bRP_Elite_68_f11.mgf",
        "Adult_Testis_bRP_Elite_68_f12.mgf",
        "Adult_Testis_bRP_Elite_68_f13.mgf",
        "Adult_Testis_bRP_Elite_68_f14.mgf",
        "Adult_Testis_bRP_Elite_68_f15.mgf",
        "Adult_Testis_bRP_Elite_68_f16.mgf",
        "Adult_Testis_bRP_Elite_68_f17.mgf",
        "Adult_Testis_bRP_Elite_68_f18.mgf",
        "Adult_Testis_bRP_Elite_68_f19.mgf",
        "Adult_Testis_bRP_Elite_68_f20.mgf",
        "Adult_Testis_bRP_Elite_68_f21.mgf",
        "Adult_Testis_bRP_Elite_68_f22.mgf",
        "Adult_Testis_bRP_Elite_68_f23.mgf",
        "Adult_Testis_bRP_Elite_68_f24.mgf"
    ]


    ADULT_URINARYBLADDER_GEL_ELITE_40_FILES = [
        "Adult_Urinarybladder_Gel_Elite_40_f01.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f02.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f03.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f04.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f05.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f06.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f07.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f08.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f09.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f10.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f11.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f12.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f13.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f14.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f15.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f16.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f17.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f18.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f19.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f20.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f21.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f22.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f23.mgf",
        "Adult_Urinarybladder_Gel_Elite_40_f24.mgf"
    ]


    ADULT_URINARYBLADDER_BRP_ELITE_71_FILES = [
        "Adult_Urinarybladder_bRP_Elite_71_f01.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f02.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f03.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f04.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f05.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f06.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f07.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f08.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f09.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f10.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f11.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f12.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f13.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f14.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f15.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f16.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f17.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f18.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f19.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f20.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f21.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f22.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f23.mgf",
        "Adult_Urinarybladder_bRP_Elite_71_f24.mgf"
    ]


    FETAL_BRAIN_BRP_ELITE_15_FILES = [
        "Fetal_Brain_bRP_Elite_15_f01.mgf",
        "Fetal_Brain_bRP_Elite_15_f02.mgf",
        "Fetal_Brain_bRP_Elite_15_f03.mgf",
        "Fetal_Brain_bRP_Elite_15_f04.mgf",
        "Fetal_Brain_bRP_Elite_15_f05.mgf",
        "Fetal_Brain_bRP_Elite_15_f06.mgf",
        "Fetal_Brain_bRP_Elite_15_f07.mgf",
        "Fetal_Brain_bRP_Elite_15_f08.mgf",
        "Fetal_Brain_bRP_Elite_15_f09.mgf",
        "Fetal_Brain_bRP_Elite_15_f10.mgf",
        "Fetal_Brain_bRP_Elite_15_f11.mgf",
        "Fetal_Brain_bRP_Elite_15_f12.mgf",
        "Fetal_Brain_bRP_Elite_15_f13.mgf",
        "Fetal_Brain_bRP_Elite_15_f14.mgf",
        "Fetal_Brain_bRP_Elite_15_f15.mgf",
        "Fetal_Brain_bRP_Elite_15_f16.mgf",
        "Fetal_Brain_bRP_Elite_15_f17.mgf",
        "Fetal_Brain_bRP_Elite_15_f18.mgf",
        "Fetal_Brain_bRP_Elite_15_f19.mgf",
        "Fetal_Brain_bRP_Elite_15_f20.mgf",
        "Fetal_Brain_bRP_Elite_15_f21.mgf",
        "Fetal_Brain_bRP_Elite_15_f22.mgf",
        "Fetal_Brain_bRP_Elite_15_f23.mgf",
        "Fetal_Brain_bRP_Elite_15_f24.mgf"
    ]


    FETAL_GUT_BRP_ELITE_17_FILES = [
        "Fetal_Gut_bRP_Elite_17_f01.mgf",
        "Fetal_Gut_bRP_Elite_17_f02.mgf",
        "Fetal_Gut_bRP_Elite_17_f03.mgf",
        "Fetal_Gut_bRP_Elite_17_f04.mgf",
        "Fetal_Gut_bRP_Elite_17_f05.mgf",
        "Fetal_Gut_bRP_Elite_17_f06.mgf",
        "Fetal_Gut_bRP_Elite_17_f07.mgf",
        "Fetal_Gut_bRP_Elite_17_f08.mgf",
        "Fetal_Gut_bRP_Elite_17_f09.mgf",
        "Fetal_Gut_bRP_Elite_17_f10.mgf",
        "Fetal_Gut_bRP_Elite_17_f11.mgf",
        "Fetal_Gut_bRP_Elite_17_f12.mgf",
        "Fetal_Gut_bRP_Elite_17_f13.mgf",
        "Fetal_Gut_bRP_Elite_17_f14.mgf",
        "Fetal_Gut_bRP_Elite_17_f15.mgf",
        "Fetal_Gut_bRP_Elite_17_f16.mgf",
        "Fetal_Gut_bRP_Elite_17_f17.mgf",
        "Fetal_Gut_bRP_Elite_17_f18.mgf",
        "Fetal_Gut_bRP_Elite_17_f19.mgf",
        "Fetal_Gut_bRP_Elite_17_f20.mgf",
        "Fetal_Gut_bRP_Elite_17_f21.mgf",
        "Fetal_Gut_bRP_Elite_17_f22.mgf",
        "Fetal_Gut_bRP_Elite_17_f23.mgf",
        "Fetal_Gut_bRP_Elite_17_f24.mgf"
    ]


    FETAL_GUT_BRP_ELITE_18_FILES = [
        "Fetal_Gut_bRP_Elite_18_f01.mgf",
        "Fetal_Gut_bRP_Elite_18_f02.mgf",
        "Fetal_Gut_bRP_Elite_18_f03.mgf",
        "Fetal_Gut_bRP_Elite_18_f04.mgf",
        "Fetal_Gut_bRP_Elite_18_f05.mgf",
        "Fetal_Gut_bRP_Elite_18_f06.mgf",
        "Fetal_Gut_bRP_Elite_18_f07.mgf",
        "Fetal_Gut_bRP_Elite_18_f08.mgf",
        "Fetal_Gut_bRP_Elite_18_f09.mgf",
        "Fetal_Gut_bRP_Elite_18_f10.mgf",
        "Fetal_Gut_bRP_Elite_18_f11.mgf",
        "Fetal_Gut_bRP_Elite_18_f12.mgf",
        "Fetal_Gut_bRP_Elite_18_f13.mgf",
        "Fetal_Gut_bRP_Elite_18_f14.mgf",
        "Fetal_Gut_bRP_Elite_18_f15.mgf",
        "Fetal_Gut_bRP_Elite_18_f16.mgf",
        "Fetal_Gut_bRP_Elite_18_f17.mgf",
        "Fetal_Gut_bRP_Elite_18_f18.mgf",
        "Fetal_Gut_bRP_Elite_18_f19.mgf",
        "Fetal_Gut_bRP_Elite_18_f20.mgf",
        "Fetal_Gut_bRP_Elite_18_f21.mgf",
        "Fetal_Gut_bRP_Elite_18_f22.mgf",
        "Fetal_Gut_bRP_Elite_18_f23.mgf",
        "Fetal_Gut_bRP_Elite_18_f24.mgf"
    ]


    FETAL_HEART_GEL_VELOS_21_FILES = [
        "Fetal_Heart_Gel_Velos_21_f01.mgf",
        "Fetal_Heart_Gel_Velos_21_f02.mgf",
        "Fetal_Heart_Gel_Velos_21_f03.mgf",
        "Fetal_Heart_Gel_Velos_21_f04.mgf",
        "Fetal_Heart_Gel_Velos_21_f05.mgf",
        "Fetal_Heart_Gel_Velos_21_f06.mgf",
        "Fetal_Heart_Gel_Velos_21_f07.mgf",
        "Fetal_Heart_Gel_Velos_21_f08.mgf",
        "Fetal_Heart_Gel_Velos_21_f09.mgf",
        "Fetal_Heart_Gel_Velos_21_f10.mgf",
        "Fetal_Heart_Gel_Velos_21_f11.mgf",
        "Fetal_Heart_Gel_Velos_21_f12.mgf",
        "Fetal_Heart_Gel_Velos_21_f13.mgf",
        "Fetal_Heart_Gel_Velos_21_f14.mgf",
        "Fetal_Heart_Gel_Velos_21_f15.mgf",
        "Fetal_Heart_Gel_Velos_21_f16.mgf",
        "Fetal_Heart_Gel_Velos_21_f17.mgf",
        "Fetal_Heart_Gel_Velos_21_f18.mgf",
        "Fetal_Heart_Gel_Velos_21_f19.mgf",
        "Fetal_Heart_Gel_Velos_21_f20.mgf",
        "Fetal_Heart_Gel_Velos_21_f21.mgf",
        "Fetal_Heart_Gel_Velos_21_f22.mgf",
        "Fetal_Heart_Gel_Velos_21_f23.mgf",
        "Fetal_Heart_Gel_Velos_21_f24.mgf",
        "Fetal_Heart_Gel_Velos_21_f25.mgf",
        "Fetal_Heart_Gel_Velos_21_f26.mgf",
        "Fetal_Heart_Gel_Velos_21_f27.mgf"
    ]


    FETAL_HEART_GEL_VELOS_73_FILES = [
        "Fetal_Heart_Gel_Velos_73_f01.mgf",
        "Fetal_Heart_Gel_Velos_73_f02.mgf",
        "Fetal_Heart_Gel_Velos_73_f03.mgf",
        "Fetal_Heart_Gel_Velos_73_f04.mgf",
        "Fetal_Heart_Gel_Velos_73_f05.mgf",
        "Fetal_Heart_Gel_Velos_73_f06.mgf",
        "Fetal_Heart_Gel_Velos_73_f07.mgf",
        "Fetal_Heart_Gel_Velos_73_f08.mgf",
        "Fetal_Heart_Gel_Velos_73_f09.mgf",
        "Fetal_Heart_Gel_Velos_73_f10.mgf",
        "Fetal_Heart_Gel_Velos_73_f11.mgf",
        "Fetal_Heart_Gel_Velos_73_f12.mgf",
        "Fetal_Heart_Gel_Velos_73_f13.mgf",
        "Fetal_Heart_Gel_Velos_73_f14.mgf",
        "Fetal_Heart_Gel_Velos_73_f15.mgf",
        "Fetal_Heart_Gel_Velos_73_f16.mgf",
        "Fetal_Heart_Gel_Velos_73_f17.mgf",
        "Fetal_Heart_Gel_Velos_73_f18.mgf",
        "Fetal_Heart_Gel_Velos_73_f19.mgf",
        "Fetal_Heart_Gel_Velos_73_f20.mgf",
        "Fetal_Heart_Gel_Velos_73_f21.mgf",
        "Fetal_Heart_Gel_Velos_73_f22.mgf",
        "Fetal_Heart_Gel_Velos_73_f23.mgf",
        "Fetal_Heart_Gel_Velos_73_f24.mgf"
    ]


    FETAL_HEART_BRP_ELITE_19_FILES = [
        "Fetal_Heart_bRP_Elite_19_f01.mgf",
        "Fetal_Heart_bRP_Elite_19_f02.mgf",
        "Fetal_Heart_bRP_Elite_19_f03.mgf",
        "Fetal_Heart_bRP_Elite_19_f04.mgf",
        "Fetal_Heart_bRP_Elite_19_f05.mgf",
        "Fetal_Heart_bRP_Elite_19_f06.mgf",
        "Fetal_Heart_bRP_Elite_19_f07.mgf",
        "Fetal_Heart_bRP_Elite_19_f08.mgf",

        #
        # Removing this file since Crux has not found any identification here
        #
        # "Fetal_Heart_bRP_Elite_19_f09.mgf",
        "Fetal_Heart_bRP_Elite_19_f10.mgf",

        #
        # Removing this file since Crux has not found any identification here
        #
        # "Fetal_Heart_bRP_Elite_19_f11.mgf",
        "Fetal_Heart_bRP_Elite_19_f12.mgf",
        "Fetal_Heart_bRP_Elite_19_f13.mgf",
        "Fetal_Heart_bRP_Elite_19_f14.mgf",
        "Fetal_Heart_bRP_Elite_19_f15.mgf",
        "Fetal_Heart_bRP_Elite_19_f16.mgf",
        "Fetal_Heart_bRP_Elite_19_f17.mgf",
        "Fetal_Heart_bRP_Elite_19_f18.mgf",
        "Fetal_Heart_bRP_Elite_19_f19.mgf",
        "Fetal_Heart_bRP_Elite_19_f20.mgf",
        "Fetal_Heart_bRP_Elite_19_f21.mgf",

        #
        # Removing this file since Crux has not found any identification here
        #
        # "Fetal_Heart_bRP_Elite_19_f22.mgf",
        "Fetal_Heart_bRP_Elite_19_f23.mgf",
        "Fetal_Heart_bRP_Elite_19_f24.mgf",
        "Fetal_Heart_bRP_Elite_19_f25.mgf",
        "Fetal_Heart_bRP_Elite_19_f26.mgf",
        "Fetal_Heart_bRP_Elite_19_f27.mgf",
        "Fetal_Heart_bRP_Elite_19_f28.mgf",
        "Fetal_Heart_bRP_Elite_19_f29.mgf"
    ]


    FETAL_HEART_BRP_ELITE_20_FILES = [
        "Fetal_Heart_bRP_Elite_20_f01.mgf",
        "Fetal_Heart_bRP_Elite_20_f02.mgf",
        "Fetal_Heart_bRP_Elite_20_f03.mgf",
        "Fetal_Heart_bRP_Elite_20_f04.mgf",
        "Fetal_Heart_bRP_Elite_20_f05.mgf",
        "Fetal_Heart_bRP_Elite_20_f06.mgf",
        "Fetal_Heart_bRP_Elite_20_f07.mgf",
        "Fetal_Heart_bRP_Elite_20_f08.mgf",
        "Fetal_Heart_bRP_Elite_20_f09.mgf",
        "Fetal_Heart_bRP_Elite_20_f10.mgf",
        "Fetal_Heart_bRP_Elite_20_f11.mgf",
        "Fetal_Heart_bRP_Elite_20_f12.mgf",
        "Fetal_Heart_bRP_Elite_20_f13.mgf",
        "Fetal_Heart_bRP_Elite_20_f14.mgf",
        "Fetal_Heart_bRP_Elite_20_f15.mgf",
        "Fetal_Heart_bRP_Elite_20_f16.mgf",
        "Fetal_Heart_bRP_Elite_20_f17.mgf",
        "Fetal_Heart_bRP_Elite_20_f18.mgf",
        "Fetal_Heart_bRP_Elite_20_f19.mgf",
        "Fetal_Heart_bRP_Elite_20_f20.mgf",
        "Fetal_Heart_bRP_Elite_20_f21.mgf",
        "Fetal_Heart_bRP_Elite_20_f22.mgf",
        "Fetal_Heart_bRP_Elite_20_f23.mgf",
        "Fetal_Heart_bRP_Elite_20_f24.mgf"
    ]


    FETAL_OVARY_GEL_VELOS_74_FILES = [
        "Fetal_Ovary_Gel_Velos_74_f01.mgf",
        "Fetal_Ovary_Gel_Velos_74_f02.mgf",
        "Fetal_Ovary_Gel_Velos_74_f03.mgf",
        "Fetal_Ovary_Gel_Velos_74_f04.mgf",
        "Fetal_Ovary_Gel_Velos_74_f05.mgf",
        "Fetal_Ovary_Gel_Velos_74_f06.mgf",
        "Fetal_Ovary_Gel_Velos_74_f07.mgf",
        "Fetal_Ovary_Gel_Velos_74_f08.mgf",
        "Fetal_Ovary_Gel_Velos_74_f09.mgf",
        "Fetal_Ovary_Gel_Velos_74_f10.mgf",
        "Fetal_Ovary_Gel_Velos_74_f11.mgf",
        "Fetal_Ovary_Gel_Velos_74_f12.mgf",
        "Fetal_Ovary_Gel_Velos_74_f13.mgf",
        "Fetal_Ovary_Gel_Velos_74_f14.mgf",
        "Fetal_Ovary_Gel_Velos_74_f15.mgf",
        "Fetal_Ovary_Gel_Velos_74_f16.mgf",
        "Fetal_Ovary_Gel_Velos_74_f17.mgf",
        "Fetal_Ovary_Gel_Velos_74_f18.mgf",
        "Fetal_Ovary_Gel_Velos_74_f19.mgf",
        "Fetal_Ovary_Gel_Velos_74_f20.mgf",
        "Fetal_Ovary_Gel_Velos_74_f21.mgf",
        "Fetal_Ovary_Gel_Velos_74_f22.mgf",
        "Fetal_Ovary_Gel_Velos_74_f23.mgf",
        "Fetal_Ovary_Gel_Velos_74_f24.mgf"
    ]


    FETAL_PLACENTA_GEL_VELOS_14_FILES = [
        "Fetal_Placenta_Gel_Velos_14_f01.mgf",
        "Fetal_Placenta_Gel_Velos_14_f02.mgf",
        "Fetal_Placenta_Gel_Velos_14_f03.mgf",
        "Fetal_Placenta_Gel_Velos_14_f04.mgf",
        "Fetal_Placenta_Gel_Velos_14_f05.mgf",
        "Fetal_Placenta_Gel_Velos_14_f06.mgf",
        "Fetal_Placenta_Gel_Velos_14_f07.mgf",
        "Fetal_Placenta_Gel_Velos_14_f08.mgf",
        "Fetal_Placenta_Gel_Velos_14_f09.mgf",
        "Fetal_Placenta_Gel_Velos_14_f10.mgf",
        "Fetal_Placenta_Gel_Velos_14_f11.mgf",
        "Fetal_Placenta_Gel_Velos_14_f12.mgf",
        "Fetal_Placenta_Gel_Velos_14_f13.mgf",
        "Fetal_Placenta_Gel_Velos_14_f14.mgf",
        "Fetal_Placenta_Gel_Velos_14_f15.mgf",
        "Fetal_Placenta_Gel_Velos_14_f16.mgf",
        "Fetal_Placenta_Gel_Velos_14_f17.mgf",
        "Fetal_Placenta_Gel_Velos_14_f18.mgf",
        "Fetal_Placenta_Gel_Velos_14_f19.mgf",
        "Fetal_Placenta_Gel_Velos_14_f20.mgf",
        "Fetal_Placenta_Gel_Velos_14_f21.mgf",
        "Fetal_Placenta_Gel_Velos_14_f22.mgf",
        "Fetal_Placenta_Gel_Velos_14_f23.mgf",
        "Fetal_Placenta_Gel_Velos_14_f24.mgf",
        "Fetal_Placenta_Gel_Velos_14_f25.mgf",
        "Fetal_Placenta_Gel_Velos_14_f26.mgf",
        "Fetal_Placenta_Gel_Velos_14_f27.mgf"
    ]


    FETAL_PLACENTA_BRP_ELITE_79_FILES = [
        "Fetal_Placenta_bRP_Elite_79_f01.mgf",
        "Fetal_Placenta_bRP_Elite_79_f02.mgf",
        "Fetal_Placenta_bRP_Elite_79_f03.mgf",
        "Fetal_Placenta_bRP_Elite_79_f04.mgf",
        "Fetal_Placenta_bRP_Elite_79_f05.mgf",
        "Fetal_Placenta_bRP_Elite_79_f06.mgf",
        "Fetal_Placenta_bRP_Elite_79_f07.mgf",
        "Fetal_Placenta_bRP_Elite_79_f08.mgf",
        "Fetal_Placenta_bRP_Elite_79_f09.mgf",
        "Fetal_Placenta_bRP_Elite_79_f10.mgf",
        "Fetal_Placenta_bRP_Elite_79_f11.mgf",
        "Fetal_Placenta_bRP_Elite_79_f12.mgf",
        "Fetal_Placenta_bRP_Elite_79_f13.mgf",
        "Fetal_Placenta_bRP_Elite_79_f14.mgf",
        "Fetal_Placenta_bRP_Elite_79_f15.mgf",
        "Fetal_Placenta_bRP_Elite_79_f16.mgf",
        "Fetal_Placenta_bRP_Elite_79_f17.mgf",
        "Fetal_Placenta_bRP_Elite_79_f18.mgf",
        "Fetal_Placenta_bRP_Elite_79_f19.mgf",
        "Fetal_Placenta_bRP_Elite_79_f20.mgf",
        "Fetal_Placenta_bRP_Elite_79_f21.mgf",
        "Fetal_Placenta_bRP_Elite_79_f22.mgf",
        "Fetal_Placenta_bRP_Elite_79_f23.mgf",
        "Fetal_Placenta_bRP_Elite_79_f24.mgf"
    ]


    FETAL_TESTIS_GEL_VELOS_27_FILES = [
        "Fetal_Testis_Gel_Velos_27_f01.mgf",
        "Fetal_Testis_Gel_Velos_27_f02.mgf",
        "Fetal_Testis_Gel_Velos_27_f03.mgf",
        "Fetal_Testis_Gel_Velos_27_f04.mgf",
        "Fetal_Testis_Gel_Velos_27_f05.mgf",
        "Fetal_Testis_Gel_Velos_27_f06.mgf",
        "Fetal_Testis_Gel_Velos_27_f07.mgf",
        "Fetal_Testis_Gel_Velos_27_f08.mgf",
        "Fetal_Testis_Gel_Velos_27_f09.mgf",
        "Fetal_Testis_Gel_Velos_27_f10.mgf",
        "Fetal_Testis_Gel_Velos_27_f11.mgf",
        "Fetal_Testis_Gel_Velos_27_f12.mgf",
        "Fetal_Testis_Gel_Velos_27_f13.mgf",
        "Fetal_Testis_Gel_Velos_27_f14.mgf",
        "Fetal_Testis_Gel_Velos_27_f15.mgf",
        "Fetal_Testis_Gel_Velos_27_f16.mgf",
        "Fetal_Testis_Gel_Velos_27_f17.mgf",
        "Fetal_Testis_Gel_Velos_27_f18.mgf",
        "Fetal_Testis_Gel_Velos_27_f19.mgf",
        "Fetal_Testis_Gel_Velos_27_f20.mgf",
        "Fetal_Testis_Gel_Velos_27_f21.mgf",
        "Fetal_Testis_Gel_Velos_27_f22.mgf",
        "Fetal_Testis_Gel_Velos_27_f23.mgf",
        "Fetal_Testis_Gel_Velos_27_f24.mgf",
        "Fetal_Testis_Gel_Velos_27_f25.mgf",
        "Fetal_Testis_Gel_Velos_27_f26.mgf",
        "Fetal_Testis_Gel_Velos_27_f27.mgf",
        "Fetal_Testis_Gel_Velos_27_f28.mgf"
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
        "Fetal_Liver_bRP_Elite_q_lt_0.01_identifications.tsv" : FETAL_LIVER_BRP_ELITE_CRUX_FILES,
        "Fetal_Liver_Gel_Velos_q_lt_0.01_identifications.tsv" : FETAL_LIVER_GEL_VELOS_CRUX_FILES,
        "Fetal_Testis_bRP_Elite_q_lt_0.01_identifications.tsv" : FETAL_TESTIS_BRP_ELITE_CRUX_FILES,

        "Adult_CD4Tcells_bRP_Elite_28_q_lt_0.01_identifications.tsv" : ADULT_CD4TCELLS_BRP_ELITE_28_FILES,
        "Adult_CD4Tcells_bRP_Velos_29_q_lt_0.01_identifications.tsv" : ADULT_CD4TCELLS_BRP_VELOS_29_FILES,
        "Adult_CD8Tcells_Gel_Velos_45_q_lt_0.01_identifications.tsv" : ADULT_CD8TCELLS_GEL_VELOS_45_FILES,
        "Adult_CD8Tcells_bRP_Elite_77_q_lt_0.01_identifications.tsv" : ADULT_CD8TCELLS_BRP_ELITE_77_FILES,
        "Adult_CD8Tcells_bRP_Velos_43_q_lt_0.01_identifications.tsv" : ADULT_CD8TCELLS_BRP_VELOS_43_FILES,
        "Adult_Colon_bRP_Elite_50_q_lt_0.01_identifications.tsv" : ADULT_COLON_BRP_ELITE_50_FILES,
        "Adult_Esophagus_bRP_Velos_3_q_lt_0.01_identifications.tsv" : ADULT_ESOPHAGUS_BRP_VELOS_3_FILES,
        "Adult_Frontalcortex_Gel_Elite_80_q_lt_0.01_identifications.tsv" : ADULT_FRONTALCORTEX_GEL_ELITE_80_FILES,
        "Adult_Frontalcortex_bRP_Elite_38_q_lt_0.01_identifications.tsv" : ADULT_FRONTALCORTEX_BRP_ELITE_38_FILES,
        "Adult_Frontalcortex_bRP_Elite_85_q_lt_0.01_identifications.tsv" : ADULT_FRONTALCORTEX_BRP_ELITE_85_FILES,
        "Adult_Gallbladder_Gel_Elite_52_q_lt_0.01_identifications.tsv" : ADULT_GALLBLADDER_GEL_ELITE_52_FILES,
        "Adult_Gallbladder_bRP_Elite_53_q_lt_0.01_identifications.tsv" : ADULT_GALLBLADDER_BRP_ELITE_53_FILES,
        "Adult_Heart_Gel_Elite_54_q_lt_0.01_identifications.tsv" : ADULT_HEART_GEL_ELITE_54_FILES,
        "Adult_Heart_Gel_Velos_7_q_lt_0.01_identifications.tsv" : ADULT_HEART_GEL_VELOS_7_FILES,
        "Adult_Heart_bRP_Elite_81_q_lt_0.01_identifications.tsv" : ADULT_HEART_BRP_ELITE_81_FILES,
        "Adult_Kidney_Gel_Elite_55_q_lt_0.01_identifications.tsv" : ADULT_KIDNEY_GEL_ELITE_55_FILES,
        "Adult_Kidney_Gel_Velos_9_q_lt_0.01_identifications.tsv" : ADULT_KIDNEY_GEL_VELOS_9_FILES,
        "Adult_Kidney_bRP_Velos_8_q_lt_0.01_identifications.tsv" : ADULT_KIDNEY_BRP_VELOS_8_FILES,
        "Adult_Liver_Gel_Elilte_83_q_lt_0.01_identifications.tsv" : ADULT_LIVER_GEL_ELILTE_83_FILES,
        "Adult_Liver_Gel_Velos_11_q_lt_0.01_identifications.tsv" : ADULT_LIVER_GEL_VELOS_11_FILES,
        "Adult_Liver_bRP_Elite_82_q_lt_0.01_identifications.tsv" : ADULT_LIVER_BRP_ELITE_82_FILES,
        "Adult_Liver_bRP_Velos_10_q_lt_0.01_identifications.tsv" : ADULT_LIVER_BRP_VELOS_10_FILES,
        "Adult_Lung_Gel_Elite_56_q_lt_0.01_identifications.tsv" : ADULT_LUNG_GEL_ELITE_56_FILES,
        "Adult_Lung_Gel_Velos_13_q_lt_0.01_identifications.tsv" : ADULT_LUNG_GEL_VELOS_13_FILES,
        "Adult_Lung_bRP_Velos_12_q_lt_0.01_identifications.tsv" : ADULT_LUNG_BRP_VELOS_12_FILES,
        "Adult_Monocytes_Gel_Velos_32_q_lt_0.01_identifications.tsv" : ADULT_MONOCYTES_GEL_VELOS_32_FILES,
        "Adult_Monocytes_bRP_Elite_33_q_lt_0.01_identifications.tsv" : ADULT_MONOCYTES_BRP_ELITE_33_FILES,
        "Adult_Monocytes_bRP_Velos_31_q_lt_0.01_identifications.tsv" : ADULT_MONOCYTES_BRP_VELOS_31_FILES,
        "Adult_NKcells_Gel_Elite_78_q_lt_0.01_identifications.tsv" : ADULT_NKCELLS_GEL_ELITE_78_FILES,
        "Adult_NKcells_Gel_Velos_47_q_lt_0.01_identifications.tsv" : ADULT_NKCELLS_GEL_VELOS_47_FILES,
        "Adult_Ovary_Gel_Elite_58_q_lt_0.01_identifications.tsv" : ADULT_OVARY_GEL_ELITE_58_FILES,
        "Adult_Ovary_bRP_Elite_57_q_lt_0.01_identifications.tsv" : ADULT_OVARY_BRP_ELITE_57_FILES,
        "Adult_Pancreas_Gel_Elite_60_q_lt_0.01_identifications.tsv" : ADULT_PANCREAS_GEL_ELITE_60_FILES,
        "Adult_Platelets_Gel_Velos_36_q_lt_0.01_identifications.tsv" : ADULT_PLATELETS_GEL_VELOS_36_FILES,
        "Adult_Platelets_bRP_Velos_35_q_lt_0.01_identifications.tsv" : ADULT_PLATELETS_BRP_VELOS_35_FILES,
        "Adult_Prostate_Gel_Elite_62_q_lt_0.01_identifications.tsv" : ADULT_PROSTATE_GEL_ELITE_62_FILES,
        "Adult_Prostate_bRP_Elite_61_q_lt_0.01_identifications.tsv" : ADULT_PROSTATE_BRP_ELITE_61_FILES,
        "Adult_Rectum_Gel_Elite_63_q_lt_0.01_identifications.tsv" : ADULT_RECTUM_GEL_ELITE_63_FILES,
        "Adult_Rectum_bRP_Elite_84_q_lt_0.01_identifications.tsv" : ADULT_RECTUM_BRP_ELITE_84_FILES,
        "Adult_Retina_Gel_Elite_65_q_lt_0.01_identifications.tsv" : ADULT_RETINA_GEL_ELITE_65_FILES,
        "Adult_Retina_Gel_Velos_5_q_lt_0.01_identifications.tsv" : ADULT_RETINA_GEL_VELOS_5_FILES,
        "Adult_Retina_bRP_Elite_64_q_lt_0.01_identifications.tsv" : ADULT_RETINA_BRP_ELITE_64_FILES,

        "Adult_Spinalcord_Gel_Elite_67_q_lt_0.01_identifications.tsv" : ADULT_SPINALCORD_GEL_ELITE_67_FILES,
        "Adult_Spinalcord_bRP_Elite_66_q_lt_0.01_identifications.tsv" : ADULT_SPINALCORD_BRP_ELITE_66_FILES,
        "Adult_Testis_Gel_Elite_69_q_lt_0.01_identifications.tsv" : ADULT_TESTIS_GEL_ELITE_69_FILES,
        "Adult_Testis_bRP_Elite_68_q_lt_0.01_identifications.tsv" : ADULT_TESTIS_BRP_ELITE_68_FILES,
        "Adult_Urinarybladder_Gel_Elite_40_q_lt_0.01_identifications.tsv" : ADULT_URINARYBLADDER_GEL_ELITE_40_FILES,
        "Adult_Urinarybladder_bRP_Elite_71_q_lt_0.01_identifications.tsv" : ADULT_URINARYBLADDER_BRP_ELITE_71_FILES,
        "Fetal_Brain_bRP_Elite_15_q_lt_0.01_identifications.tsv" : FETAL_BRAIN_BRP_ELITE_15_FILES,
        "Fetal_Gut_bRP_Elite_17_q_lt_0.01_identifications.tsv" : FETAL_GUT_BRP_ELITE_17_FILES,
        "Fetal_Gut_bRP_Elite_18_q_lt_0.01_identifications.tsv" : FETAL_GUT_BRP_ELITE_18_FILES,
        "Fetal_Heart_Gel_Velos_21_q_lt_0.01_identifications.tsv" : FETAL_HEART_GEL_VELOS_21_FILES,
        "Fetal_Heart_Gel_Velos_73_q_lt_0.01_identifications.tsv" : FETAL_HEART_GEL_VELOS_73_FILES,
        "Fetal_Heart_bRP_Elite_19_q_lt_0.01_identifications.tsv" : FETAL_HEART_BRP_ELITE_19_FILES,
        "Fetal_Heart_bRP_Elite_20_q_lt_0.01_identifications.tsv" : FETAL_HEART_BRP_ELITE_20_FILES,
        "Fetal_Ovary_Gel_Velos_74_q_lt_0.01_identifications.tsv" : FETAL_OVARY_GEL_VELOS_74_FILES,
        "Fetal_Placenta_Gel_Velos_14_q_lt_0.01_identifications.tsv" : FETAL_PLACENTA_GEL_VELOS_14_FILES,
        "Fetal_Placenta_bRP_Elite_79_q_lt_0.01_identifications.tsv" : FETAL_PLACENTA_BRP_ELITE_79_FILES,
        "Fetal_Testis_Gel_Velos_27_q_lt_0.01_identifications.tsv" : FETAL_TESTIS_GEL_VELOS_27_FILES,

        #
        # q < 0.001 identifications
        #

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



    def __init__(self, identificationsFilename = None, spectraFilename = None, cruxIdentifications = False, maxPvalue = None):
    
        self.identificationsFilename = identificationsFilename
        self.spectraFilename = spectraFilename
        self.cruxIdentifications = cruxIdentifications
    
        self.maxPvalue = maxPvalue

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


            # Check if has to filter using maximun pvalue threshold

            if self.maxPvalue:
                self.uniqueCombination = self.uniqueCombination[self.uniqueCombination['pvalue'] <= self.maxPvalue]

                if self.uniqueCombination.shape[0] == 0:
                    Logger()("{} identifications file has no pvalue within the specified max pvalue threhsold ― {})".format(self.identificationsFilename,
                                                                                                                            self.maxPvalue))
                else:
                    Logger()("Identification file {} has {} spectra within the specified max pvalue threshold ― {}".format(self.identificationsFilename,
                                                                                                                           self.uniqueCombination.shape[0],
                                                                                                                           self.maxPvalue))

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




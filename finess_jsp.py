import os
import sys
import json
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

def chunk_str(str, chunk_size):
	return [str[i:i+chunk_size] for i in range(0, len(str), chunk_size)]

def Str2Int(str, Q):
	num = int(str,16)
	if num >= 2**(Q-1):
		num = num - 2**Q
	return num

def AccDecode(str):
	accQ14 = Str2Int(str,16)
	acc = float(accQ14)/2**14
	return acc

def get_acc_list(jsonFmt):
	AccRawEncoded = jsonFmt['Acc_Raw']
	AccRaw = chunk_str(AccRawEncoded, 4)
	size = len(AccRaw)
	acctemp = [0,0,0]
	acc_list = np.array([])
	ts = 0.0
	t_unit = 1.0/30
	for i in range(0,size-1):
		acctemp[i%3] = AccDecode(AccRaw[i])
		if i%3 == 2:
			acctemp.insert(0,ts)
			npacc = np.array(acctemp)
			acc_list = np.vstack([acc_list, npacc]) if acc_list.size else npacc
			acctemp = [0,0,0]
			ts = ts + t_unit
	return acc_list

class User:
	def __init__(self, Name, Age, Weight, Height, Gender):
		self.name = Name
		self.age = Age
		self.weight = Weight
		self.height = Height
		self.gender = Gender

class Result:
	def __init__(self, BPM, Time, Distance, Calorie, Calorie_hr, Steps):
		self.speed_bpm = float(BPM)
		self.elapsed_time = Time
		self.distance = float(Distance)
		self.m_calorie = float(Calorie)
		self.ref_calorie = float(Calorie_hr)
		self.m_steps = float(Steps)
		self.ref_steps = self.elapsed_time*self.speed_bpm/60.0
		self.step_length = float( self.distance/self.ref_steps )
		self.step_err_rate = float( (self.m_steps-self.ref_steps)/self.ref_steps )
		self.calorie_err_rate = float( (self.m_calorie-self.ref_calorie)/self.ref_calorie )

class Test_Report:
	def __init__(self, acc, user, result):
		self.acc_list = acc
		self.user = user
		self.result = result

def get_user_data(jsonFmt):
	jsonFmt = json.loads(jsonFmt['User'])
	user_data = User( jsonFmt['Name'],
				jsonFmt['Age'],
				jsonFmt['Weight'],
				jsonFmt['Height'],
				jsonFmt['Gender'])
	return user_data

def get_result(jsonFmt):
	jsonFmt = json.loads(jsonFmt['Result'])
	result = Result( jsonFmt['BPM'],
				jsonFmt['Time'],
				jsonFmt['Distance'],
				jsonFmt['Calorie'],
				jsonFmt['Calorie_hr'],
				jsonFmt['Steps'])
	return result

f_names = sys.argv
f_names = f_names[1:]

test_report = []

for name in f_names:
	f = open(name,'r')
	jsonRaw = f.read()
	#----------------------------------------------#
	jsonFmt = json.loads(jsonRaw)
	acc = get_acc_list(jsonFmt)
	user = get_user_data(jsonFmt)
	result = get_result(jsonFmt)
	test_report.append(Test_Report(acc, user, result))
i = 0

for test in test_report:
	u = test.user
	r = test.result
	a = test.acc_list
	print ', '.join("%s: %s" % item for item in vars(u).items())
	print ', '.join("%s: %s" % item for item in vars(r).items())

	plt.figure()
	plt.plot(a[:,0], a[:,1],'b')
	plt.plot(a[:,0], a[:,2],'r')
	plt.plot(a[:,0], a[:,3],'g')
	name = "raw{}.png".format(i)
	plt.savefig(name)
	i = i + 1

#plt.show()

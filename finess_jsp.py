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

def get_user(jsonFmt):
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

def plot_layout_save(ax, title, xlabel, ylabel, out_name):
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1.))
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.savefig(out_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

def output_inaccuarte_step_acc(test_report):
	for test in test_report:
		r = test.result
		u = test.user
		acc = test.acc_list
		if np.abs(r.step_err_rate) > 0.1:
			f_dir = './bad_step/'
			f_name = "{}acc_by_{}_at_{}_bpm".format(f_dir, u.name[0:4],int(r.speed_bpm))
			if not os.path.exists(f_dir):
				os.makedirs(f_dir)
			f = open(f_name, 'w+')
			temp = 'Freq;30Hz;\n'
			f.write(temp)
			temp = 'Measure Step;{};\n'.format(r.m_steps)
			f.write(temp)
			temp = 'Reference Step;{};\n'.format(r.ref_steps)
			f.write(temp)
			temp = 'Acc;accx;accy;accz\n'
			f.write(temp)
			for a in acc:
				a.tolist() #np.array to list
				temp = 'Acc;{1};{2};{3}\n'.format(*a)
				f.write(temp)
	return

def output_step_length_training_table(test_repost):
	f_name = 'training_table_SL'
	f = open(f_name, 'w+')
	temp = 'speed height steplength age gender\n'
	f.write(temp)
	for test in test_report:
		u = test.user
		r = test.result
		temp = "{} {} {} {} {}\n".format(int(r.speed_bpm), \
						u.height, r.step_length, u.age, u.gender)
		f.write(temp)
	return

def output_step_result(test_report):
	#[[bpm, m_steps, ref_steps, step_err_rate],...]
	step_r = np.array([])
	for test in test_report:
		r = test.result
		temp = [r.speed_bpm, r.m_steps, r.ref_steps, r.step_err_rate]
		temp = np.array(temp)
		step_r =  np.vstack([step_r, temp]) if step_r.size else temp
	step_r.sort(axis=0)
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(step_r[:,0], step_r[:,1], '*-', label='m_steps')
	ax.plot(step_r[:,0], step_r[:,2], '*-', label='ref_steps')
	ax.plot(step_r[:,0], step_r[:,1] - step_r[:,2], '*-',label='diff_steps')
	plot_layout_save(ax, 'speed vs steps', 'speed (bpm)', 'steps', "speed_vs_steps.png")
	plt.figure()
	ax = plt.subplot(111)
	ax.set_ylim([-0.15, 0.15])
	ax.plot(step_r[:,0], step_r[:,3], '*-', label='step_err_rate')
	ax.plot(step_r[:,0], np.ones(step_r[:,0].size)*0.1, label='upper bund')
	ax.plot(step_r[:,0], np.ones(step_r[:,0].size)*-0.1, label='lower bund')
	plot_layout_save(ax, 'speed vs step err rate', 'speed (bpm)', 'step err rate', "speed_vs_step_err_rate.png")
	#plt.show()
	return step_r

def get_algorithm_best_calorie(test):
	u = test.user
	r = test.result
	vel = r.distance / (r.elapsed_time / 60)
	if vel <= 134:
		METS = 0.0002575*(vel**2) + 0.0326*vel + 0.18142
	else:
		METS = 0.0577*vel - 0.0847
	cal = METS*u.weight*(r.elapsed_time/3600)
	return cal

def output_calorie_result(test_report):
	#[[bpm, m_calorie, ref_calorie, a_calorie, calorie_err_m_a, calorie_err_a_ref, ],...]
	cal_r = np.array([])
	for test in test_report:
		r = test.result
		a_calorie = get_algorithm_best_calorie(test)
		calorie_err_m_a = (r.m_calorie - a_calorie) / a_calorie
		calorie_err_a_ref = (a_calorie - r.ref_calorie) / r.ref_calorie
		temp = [r.speed_bpm, r.m_calorie, r.ref_calorie, a_calorie,\
			calorie_err_m_a, calorie_err_a_ref]
		temp = np.array(temp)
		cal_r =  np.vstack([cal_r, temp]) if cal_r.size else temp
	cal_r.sort(axis=0)
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(cal_r[:,0], cal_r[:,1], '*-', label='m_steps')
	ax.plot(cal_r[:,0], cal_r[:,2], '*-', label='ref_steps')
	plot_layout_save(ax, 'speed vs calorie', 'speed (bpm)', 'kcal', 'speed_vs_calorie.png')
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(cal_r[:,0], cal_r[:,4], '*-', label='calorie_err_m_a')
	ax.plot(cal_r[:,0], cal_r[:,5], '*-', label='calorie_err_a_ref')
	plot_layout_save(ax, 'speed vs cal err rate', 'speed (bpm)', 'kcal', 'speed_vs_cal_err_rate.png')
	#plt.show()
	return

def print_all_data(test_report):
	for test in test_report:
		u = test.user
		r = test.result
		a = test.acc_list
		print ', '.join("%s: %s" % item for item in vars(u).items())
		print ', '.join("%s: %s" % item for item in vars(r).items())

		plt.figure()
		plt.plot(a[:,0], a[:,1],'b', label='accx')
		plt.plot(a[:,0], a[:,2],'r', label='accy')
		plt.plot(a[:,0], a[:,3],'g', label='accz')
		title = "acc_by_{}_at_{}_bpm".format(u.name[0:4],r.speed_bpm)
		plt.title(title)
		plt.xlabel('timestamp')
		plt.ylabel('g')
		plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
		name = "{}.png".format(title)
		plt.savefig(name)

	#plt.show()


f_names = sys.argv
f_names = f_names[1:]

test_report = []

for name in f_names:
	f = open(name,'r')
	jsonRaw = f.read()
	#----------------------------------------------#
	jsonFmt = json.loads(jsonRaw)
	acc = get_acc_list(jsonFmt)
	user = get_user(jsonFmt)
	result = get_result(jsonFmt)
	test_report.append(Test_Report(acc, user, result))

#print_all_data(test_report)
output_step_result(test_report)
output_calorie_result(test_report)
output_inaccuarte_step_acc(test_report)
output_step_length_training_table(test_report)

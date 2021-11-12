import os
import numpy as np
import csv
from scipy.special import binom

DCM_PATH = 'D:\\PhD\\Data\\VTT'
PATH_TO_RESULTS_FILE = 'D:\\PhD\\ICM\\xnat_upload\\my_VTT_results.csv'

class VTT_results():
    
    def __init__(self):
        
        self.tasks = ['omentum', 'pelvic_ovarian']
        self.true_labels = {}
        
        for task in self.tasks:
            taskp = os.path.join(DCM_PATH, task)
            for scan in os.listdir(taskp):
                dcmrt_file = [dcm for dcm in os.listdir(os.path.join(taskp, scan)) if dcm.startswith('TCGA')][0]
                
                if dcmrt_file.endswith('RW.dcm'):
                    self.true_labels[scan+'_CT_'+task] = 'MANUAL'
                else:
                    self.true_labels[scan+'_CT_'+task] = 'AUTOMATED'

        self.all_results = []
        
        with open(PATH_TO_RESULTS_FILE, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for line_count, row in enumerate(csv_reader):
                if line_count > 0:
                    self.all_results.append(row)

    def get_reader_results(self, reader):
        
        if isinstance(reader, str):
            reader = [reader]
        
        assert isinstance(reader, list), 'input must be str for single reader or list of readers'
        
        reader_results = {task: [] for task in self.tasks}
        
        for task in self.tasks:
            
            for result in self.all_results:
                
                if result['Reader'] in reader and result['Experiment'].endswith(task):
                    
                    reader_result = result.copy()
                    reader_result['TrueLabel'] = self.true_labels[result['Experiment']]
                    reader_result['CorrectAssessment'] = reader_result['Assessment'] == reader_result['TrueLabel']
                    reader_results[task].append(reader_result)

        for task in self.tasks:
            print('Got {} results for task {}'.format(len(reader_results[task]), task))

        return reader_results

    def compute_p_value(self, n_total, n_correct):
        
        probabilities = 0.5**n_total * np.array([binom(n_total, i) for i in range(n_total +1)])

        n1, n2 = np.min([n_correct, n_total - n_correct]), np.max([n_correct, n_total - n_correct])
        
        return np.sum(probabilities[:n1]) + np.sum(probabilities[n2+1:])

    def compute_task_metrics(self, reader_results):
        
        task_metrics = {task:{'n_correct':0} for task in self.tasks}

        for task in self.tasks:
            task_results = reader_results[task]
            
            for result in task_results:
                if result['CorrectAssessment']:
                    task_metrics[task]['n_correct'] += 1
                
            task_metrics[task]['n_total'] = len(task_results)
            task_metrics[task]['p_correct'] = 100 * task_metrics[task]['n_correct'] / task_metrics[task]['n_total']
            task_metrics[task]['p_H0'] = self.compute_p_value(task_metrics[task]['n_total'],
                                                              task_metrics[task]['n_correct'])
        
            print(task+': ')
            for key in task_metrics[task]:
                print('\t{}: {:.3f}'.format(key, task_metrics[task][key]))
        
        return task_metrics
        
results = VTT_results()
reader_results = results.get_reader_results('vtt_internal2')
task_metrics = results.compute_task_metrics(reader_results)

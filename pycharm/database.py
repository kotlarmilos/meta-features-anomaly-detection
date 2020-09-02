import mysql.connector
import pandas as pd
import numpy as np

class Database:
    def __init__(self, host, user, passwd, database):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database

    def open_connection(self):
        self.db = mysql.connector.connect(user=self.user, password=self.passwd,
                              host=self.host,
                              database=self.database)
        self.cursor = self.db.cursor()

    def close_connection(self):
        self.cursor.close()
        self.db.close()

    def truncate_database(self):
        self.open_connection()
        self.cursor.execute('DELETE FROM algorithm')
        self.db.commit()
        sql = "INSERT INTO algorithm (name, complexity) VALUES (%s, %s)"
        val = ('Gaussian', 'O(n^2)')
        self.cursor.execute(sql, val)
        val = ('Linear', 'O(n^2)')
        self.cursor.execute(sql, val)
        val = ('RPCA', 'O(n^2)')
        self.cursor.execute(sql, val)
        val = ('KMeans', 'O(n^2)')
        self.cursor.execute(sql, val)
        val = ('AutoencoderModel', 'O(n^2)')
        self.cursor.execute(sql, val)

        self.db.commit()

        self.cursor.execute('DELETE FROM device_characterization')
        self.cursor.execute('DELETE FROM device')

        sql = "INSERT INTO device (name, type) VALUES (%s, %s)"
        val = ('Intel Xeon', 'CPU')
        self.cursor.execute(sql, val)

        device_id = self.cursor.lastrowid
        sql = "INSERT INTO device_characterization (device_id, name, value) VALUES (%s, %s, %s)"
        val = (device_id, 'transistor_count', '7.2')
        self.cursor.execute(sql, val)
        val = (device_id, 'core_count', '2')
        self.cursor.execute(sql, val)
        val = (device_id, 'technology', '22')
        self.cursor.execute(sql, val)
        val = (device_id, 'power_dissipation', '180')
        self.cursor.execute(sql, val)
        val = (device_id, 'flops', '90')
        self.cursor.execute(sql, val)
        val = (device_id, 'fequency', '4300')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_type', 'DRAM')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_size', '13')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_bandwidth', '')
        self.cursor.execute(sql, val)
        val = (device_id, 'weight', '')
        self.cursor.execute(sql, val)

        sql = "INSERT INTO device (name, type) VALUES (%s, %s)"
        val = ('Tesla K80', 'GPU')
        self.cursor.execute(sql, val)
        device_id = self.cursor.lastrowid
        sql = "INSERT INTO device_characterization (device_id, name, value) VALUES (%s, %s, %s)"
        val = (device_id, 'transistor_count', '7.1')
        self.cursor.execute(sql, val)
        val = (device_id, 'core_count', '2496')
        self.cursor.execute(sql, val)
        val = (device_id, 'technology', '28')
        self.cursor.execute(sql, val)
        val = (device_id, 'power_dissipation', '300')
        self.cursor.execute(sql, val)
        val = (device_id, 'flops', '2910')
        self.cursor.execute(sql, val)
        val = (device_id, 'fequency', '1562')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_type', 'DRAM')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_size', '12')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_bandwidth', '')
        self.cursor.execute(sql, val)
        val = (device_id, 'weight', '')
        self.cursor.execute(sql, val)

        sql = "INSERT INTO device (name, type) VALUES (%s, %s)"
        val = ('Google TPU', 'ASIC')
        self.cursor.execute(sql, val)
        device_id = self.cursor.lastrowid
        sql = "INSERT INTO device_characterization (device_id, name, value) VALUES (%s, %s, %s)"
        val = (device_id, 'transistor_count', '2.1')
        self.cursor.execute(sql, val)
        val = (device_id, 'core_count', '2496')
        self.cursor.execute(sql, val)
        val = (device_id, 'technology', '28')
        self.cursor.execute(sql, val)
        val = (device_id, 'power_dissipation', '40')
        self.cursor.execute(sql, val)
        val = (device_id, 'flops', '180000')
        self.cursor.execute(sql, val)
        val = (device_id, 'fequency', '700')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_type', 'SRAM')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_size', '16')
        self.cursor.execute(sql, val)
        val = (device_id, 'memory_bandwidth', '')
        self.cursor.execute(sql, val)
        val = (device_id, 'weight', '')
        self.cursor.execute(sql, val)
        self.db.commit()

        self.cursor.execute('DELETE FROM dataset_characterization')
        self.cursor.execute('DELETE FROM feature_score')
        self.cursor.execute('DELETE FROM dataset')

        self.cursor.execute('DELETE FROM performance')
        self.cursor.execute('DELETE FROM parameter')
        self.cursor.execute('DELETE FROM evaluation')

        self.db.commit()

        self.close_connection()

    def get_dataset_id(self, dataset):
        self.open_connection()
        self.cursor.execute("SELECT * FROM dataset WHERE name='" + str(dataset['name']) + "'")
        row = self.cursor.fetchone()
        if row:
            return row[0]
        else:
            return None

    def get_dataset_anomaly_ratio(self, dataset):
        self.open_connection()
        self.cursor.execute("SELECT anomaly_entropy FROM dataset WHERE name='" + str(dataset['name']) + "'")
        row = self.cursor.fetchone()
        if row:
            return row[0]
        else:
            return None

    def get_datasets(self):
        self.open_connection()
        self.cursor.execute("SELECT * FROM dataset")
        dataset = pd.DataFrame(self.cursor.fetchall())
        dataset.columns = np.array([i[0] for i in self.cursor.description])
        # dataset = dataset.set_index('id')

        self.cursor.execute("""select e.training_time, a.name as method, d.id as dataset_id, d.name as dataset, 
                            (select cast(value as decimal(24,8)) from performance p where p.evaluation_id = e.id and name = 'acc' limit 1) as acc,
                            (select cast(value as decimal(24,8)) from performance p where p.evaluation_id = e.id and name = 'prec' limit 1) as prec,
                            (select cast(value as decimal(24,8)) from performance p where p.evaluation_id = e.id and name = 'recall' limit 1) as recall,
                            (select cast(value as decimal(24,8)) from performance p where p.evaluation_id = e.id and name = 'f1' limit 1) as f1,
                            (select cast(value as decimal(24,8)) from parameter p where p.evaluation_id = e.id and name = 'pca' limit 1) as pca,
                            (select cast(value as decimal(24,8)) from parameter p where p.evaluation_id = e.id and name = 'k' limit 1) as k
                            from evaluation e
                            left join algorithm a on e.algorithm_id = a.id
                            left join dataset d on e.dataset_id = d.id""")
        evaluation = pd.DataFrame(self.cursor.fetchall(), dtype='float64')
        evaluation.columns = np.array([i[0] for i in self.cursor.description])

        self.cursor.execute("SELECT dataset_id, name, cast(case when value = 'nan' then 0 else value end as decimal(24,8)) as value FROM dataset_characterization")
        features = pd.DataFrame(self.cursor.fetchall(), dtype='float64')
        features.columns = [i[0] for i in self.cursor.description]
        features = features.pivot(index='dataset_id', columns='name', values='value')
        # result = pd.concat([df, df_ft], axis=1, sort=False)
        return dataset, evaluation, features

    def insert_data_info(self, dataset, ft, feature_score = None):
        self.open_connection()
        self.cursor.execute("SELECT * FROM dataset WHERE name='" + str(dataset['name']) + "'")
        row = self.cursor.fetchone()
        if row:
            return row[0]

        sql = "INSERT INTO dataset (name, type_of_data, domain, anomaly_types, anomaly_space, anomaly_entropy, label, files) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (dataset['name'], str(dataset['type_of_data']).strip('[]'), str(dataset['domain']).strip('[]'),
               str(dataset['anomaly_types']).strip('[]'), dataset['anomaly_space'], dataset['anomaly_entropy'],
               dataset['label'], str(dataset['files']).strip('[]'))
        self.cursor.execute(sql, val)
        self.db.commit()
        dataset['id'] = self.cursor.lastrowid

        if len(ft) == 2:
            for i in range(len(ft[0])):
                sql = "INSERT INTO dataset_characterization (dataset_id, name, value) VALUES (%s, %s, %s)"
                val = (dataset['id'], str(ft[0][i]), str(ft[1][i]))
                self.cursor.execute(sql, val)
            self.db.commit()

        if feature_score:
            for key in feature_score:
                sql = "INSERT INTO feature_score (dataset_id, name, value) VALUES (%s, %s, %s)"
                val = (dataset['id'], str(key), str(feature_score[key]))
                self.cursor.execute(sql, val)
            self.db.commit()

        self.close_connection()
        return dataset['id']

    def check_evaluation_info(self, device, method,dataset, p, headers):
        device_type = 'ASIC'
        if 'CPU' in device:
            device_type = 'CPU'
        elif 'GPU' in device:
            device_type = 'GPU'

        self.open_connection()
        self.cursor.execute("SELECT * FROM device WHERE type='" + device_type + "'")
        device_id = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT * FROM dataset WHERE id=" + str(dataset['id']))
        dataset_id = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT * FROM algorithm WHERE name='" + method['name'] + "'")
        algorithm_id = self.cursor.fetchone()[0]

        if len(p) > 1:
            sql = "select * from evaluation e where dataset_id=%s and algorithm_id=%s and device_id=%s and exists (select 1 from parameter p where p.evaluation_id=e.id and ((p.name='pca' and p.value=%s))) and exists (select 1 from parameter p where p.evaluation_id=e.id and ((p.name='k' and p.value=%s)))"
            val = (dataset_id, algorithm_id, device_id, str(p[0]), str(p[1]))
        else:
            sql = "select * from evaluation e where dataset_id=%s and algorithm_id=%s and device_id=%s and exists (select 1 from parameter p where p.evaluation_id=e.id and ((p.name='pca' and p.value=%s)))"
            val = (dataset_id, algorithm_id, device_id, str(p[0]))


        self.cursor.execute(sql, val)
        rows = self.cursor.fetchall()
        result = len(rows)
        self.close_connection()
        if result > 0:
            return True
        else:
            return False

    def insert_evaluation_info(self, device, method, dataset, params, headers, time, result):
        device_type = 'ASIC'
        if 'CPU' in device:
            device_type = 'CPU'
        elif 'GPU' in device:
            device_type = 'GPU'

        self.open_connection()
        self.cursor.execute("SELECT * FROM device WHERE type='" + device_type + "'")
        device_id = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT * FROM dataset WHERE id=" + str(dataset['id']))
        dataset_id = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT * FROM algorithm WHERE name='" + method['name'] + "'")
        algorithm_id = self.cursor.fetchone()[0]

        sql = "INSERT INTO evaluation (evaluation_id, dataset_id, algorithm_id, device_id, training_time, inference_time) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (1, dataset_id, algorithm_id, device_id, str(time), str(time))
        self.cursor.execute(sql, val)
        self.db.commit()

        evaluation_id = self.cursor.lastrowid
        sql = "INSERT INTO performance (evaluation_id, name, value) VALUES (%s, %s, %s)"
        val = (evaluation_id, 'acc', str(result['acc']['scores']['acc']))
        self.cursor.execute(sql, val)
        val = (evaluation_id, 'prec', str(result['prec']['scores']['prec']))
        self.cursor.execute(sql, val)
        val = (evaluation_id, 'recall', str(result['recall']['scores']['recall']))
        self.cursor.execute(sql, val)
        val = (evaluation_id, 'f1', str(result['f1']['scores']['f1']))
        self.cursor.execute(sql, val)
        val = (evaluation_id, 'manual', str(result['f1']['scores']['f1']))
        self.cursor.execute(sql, val)
        self.db.commit()

        sql = "INSERT INTO parameter (evaluation_id, name, value) VALUES (%s, %s, %s)"
        for i in range(len(params)):
            val = (evaluation_id, headers[i], str(params[i]))
            self.cursor.execute(sql, val)
        self.db.commit()

        self.close_connection()

    def update_characterization_user_defined_data(self, dataset):
        dataset['id'] = self.get_dataset_id(dataset)
        type_of_data = ['nominal', 'spatial', 'temporal', 'graphs_and_networks', 'high_dimensional']
        domain = ['manufacturing', 'transport', 'finance', 'medicine', 'images', 'text', 'software', 'social']
        anomaly_type = ['local', 'global', 'cluster']

        for t in type_of_data:
            sql = "DELETE FROM dataset_characterization where dataset_id=%s and name=%s"
            val = (dataset['id'], t)
            self.cursor.execute(sql, val)
            self.db.commit()
            sql = "INSERT INTO dataset_characterization (dataset_id, name, value) VALUES (%s, %s, %s)"
            val = (dataset['id'], t, str(1 if (t in dataset['type_of_data']) else 0))
            self.cursor.execute(sql, val)
            self.db.commit()

        for d in domain:
            sql = "DELETE FROM dataset_characterization where dataset_id=%s and name=%s"
            val = (dataset['id'], d)
            self.cursor.execute(sql, val)
            sql = "INSERT INTO dataset_characterization (dataset_id, name, value) VALUES (%s, %s, %s)"
            val = (dataset['id'], d, str(1 if (d in dataset['domain']) else 0))
            self.cursor.execute(sql, val)

        for a in anomaly_type:
            sql = "DELETE FROM dataset_characterization where dataset_id=%s and name=%s"
            val = (dataset['id'], a)
            self.cursor.execute(sql, val)
            self.db.commit()
            sql = "INSERT INTO dataset_characterization (dataset_id, name, value) VALUES (%s, %s, %s)"
            val = (dataset['id'], a, str(1 if (a in dataset['anomaly_types']) else 0))
            self.cursor.execute(sql, val)
            self.db.commit()


        sql = "DELETE FROM dataset_characterization where dataset_id=%s and name=%s"
        val = (dataset['id'], 'anomaly_space')
        self.cursor.execute(sql, val)
        self.db.commit()
        sql = "INSERT INTO dataset_characterization (dataset_id, name, value) VALUES (%s, %s, %s)"
        val = (dataset['id'], 'anomaly_space', str(0 if dataset['anomaly_space'] == 'univariate' else 1))
        self.cursor.execute(sql, val)
        self.db.commit()

        sql = "DELETE FROM dataset_characterization where dataset_id=%s and name=%s"
        val = (dataset['id'], 'anomaly_ratio')
        self.cursor.execute(sql, val)
        self.db.commit()
        anomaly_ratio = self.get_dataset_anomaly_ratio(dataset)
        sql = "INSERT INTO dataset_characterization (dataset_id, name, value) VALUES (%s, %s, %s)"
        val = (dataset['id'], 'anomaly_ratio', anomaly_ratio)
        self.cursor.execute(sql, val)
        self.db.commit()

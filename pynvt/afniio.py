import re
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.stats as stats
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN


class AFNIIO():

    def __init__(self, path, **kwargs):

        # kwargs
        self.NN = kwargs["NN"] if "NN" in kwargs else 2
        self.thr_p = kwargs["thr_p"] if "thr_p" in kwargs else 0.05
        self.thr_a = kwargs["thr_a"] if "thr_a" in kwargs else 0.01

        # Set
        self.scene_data = dict()
        self.dataset_rank = dict()
        self.brick_statsymbol = dict()
        self.brick_label = dict()
        self.clustsim_table = dict()
        self.clustsim_ref = dict()
        self.max_overlay = 0
        self.min_overlay = 0

        # LOAD
        self.nii = nib.load(path)

        # Parse
        self.parse()

    def parse(self):

        # pattern to parse attributes and contents in clustsim_tag
        p1 = r'^"<3dClustSim_\w{2}\d{1}' \
             '(?P<attributes>(?!>).*)\>' \
             '(?P<contents>(?!<).*)' \
             '</3dClustSim_\w{2}\d{1}>\"'
        # pattern to parse clustsim condition
        p2 = r'^AFNI_CLUSTSIM_(?P<nn>NN\d{1})_(?P<side>\w+sided)$'
        # pattern to parse dictionary stricture
        p3 = r'((?!=")[a-z_]*)\=\"(.+?)\"'
        # pattern to parse sub-brick stats
        p4 = r'^(?P<test>Ttest|Ftest|Zscore)\((?P<param>.*)\)'

        view_type = {0: '+orig', 1: '+acpc', 2: '+tlrc'}

        anat_type = {0: 'ANAT_SPGR',
                     1: 'ANAT_FSE',
                     2: 'ANAT_EPI',
                     3: 'ANAT_MRAN',
                     4: 'ANAT_CT',
                     5: 'ANAT_SPECT',
                     6: 'ANAT_PET',
                     7: 'ANAT_MRA',
                     8: 'ANAT_BMAP',
                     9: 'ANAT_DIFF',
                     10: 'ANAT_OMRI',
                     11: 'ANAT_BUCK'
                     }

        func_type = {0: 'FUNC_FIM',  # 1 value
                     1: 'FUNC_THR',  # obsolete
                     2: 'FUNC_COR',  # fico: correlation
                     3: 'FUNC_TT',  # fitt: t-statistic
                     4: 'FUNC_FT',  # fift: F-statistic
                     5: 'FUNC_ZT',  # fizt: z-score
                     6: 'FUNC_CT',  # fict: Chi squared
                     7: 'FUNC_BT',  # fibt: Beta stat
                     8: 'FUNC_BN',  # fibn: Binomial
                     9: 'FUNC_GT',  # figt: Gamma
                     10: 'FUNC_PT',  # fipt: Poisson
                     11: 'FUNC_BUCK'  # fbuc: bucket
                     }

        type_string = {0: "3DIM_HEAD_ANAT",
                       1: "3DIM_HEAD_FUNC",
                       2: "3DIM_GEN_ANAT",
                       3: "3DIM_GEN_FUNC"
                       }

        for ext in self.nii.header.extensions:
            root = ET.fromstring(ext.get_content())
            if root.tag == 'AFNI_attributes':
                for child in root:
                    if re.match('^SCENE_DATA$', child.attrib['atr_name']):
                        # parse SCENE_DATA
                        scene_data = np.asarray(child.text.strip().split('\n'), dtype=int)[:3]
                        v, tp, ts = scene_data
                        self.scene_data['view_type'] = view_type[v]
                        if 'ANAT' in type_string[ts]:
                            self.scene_data['data_type'] = anat_type[tp]
                        else:
                            self.scene_data['data_type'] = func_type[tp]

                    elif re.match('^DATASET_RANK$', child.attrib['atr_name']):
                        # parse DATASET_RANK
                        dataset_rank = np.asarray(child.text.strip().split('\n'), dtype=int)[:2]
                        self.dataset_rank['spatial_dim'] = dataset_rank[0]
                        self.dataset_rank['num_sub-bricks'] = dataset_rank[1]

                    elif re.match('^BRICK_STATSYM', child.attrib['atr_name']):
                        # parse BRICK_STATSYM - contains info which statistics uses for each sub-bricks
                        brick_stats = child.text.strip().strip('"')
                        stats_type = brick_stats.split(';')
                        for i, tp in enumerate(stats_type):
                            if tp == 'none':
                                self.brick_statsymbol[i] = None
                            else:
                                test_type = re.sub(p4, r'\g<test>', tp)
                                param = re.sub(p4, r'\g<param>', tp)
                                if len(param) == 0:
                                    param = None
                                elif len(param.split(',')) > 1:
                                    param = list(map(int, param.split(',')))
                                elif test_type == 'none':
                                    test_type = None
                                    param = None
                                else:
                                    param = int(param)
                                if test_type is None and param is None:
                                    self.brick_statsymbol[i] = None
                                else:
                                    self.brick_statsymbol[i] = [test_type, param]

                    elif re.match('^BRICK_LAB', child.attrib['atr_name']):
                        # parse BRICK_LAB
                        brick_lab = child.text.strip().strip('"')
                        self.brick_label = {i: t for i, t in enumerate(brick_lab.split('~'))}

                    elif re.match(p2, child.attrib['atr_name']):
                        # parse ClustSim tables
                        clustsim = re.sub(p2, r'\g<nn>-\g<side>', child.attrib['atr_name'])
                        text = ''.join(child.text.split('\n')).strip()
                        if re.match(p1, text):
                            cont = ''.join(re.sub(p1, r'\g<contents>', text).split('" "')).strip().split(' ')
                            if 'p' not in self.clustsim_ref.keys():
                                attr = re.sub(p1, r'\g<attributes>', text)
                                attr = dict(re.findall(p3, attr))
                                p = list(map(float, attr['pthr'].split(',')))
                                self.clustsim_ref['p'] = p
                                a = list(map(float, attr['athr'].split(',')))
                                self.clustsim_ref['a'] = a
                            csim_table = np.asarray(cont, dtype='float').reshape([len(p), len(a)])
                            self.clustsim_table[clustsim] = pd.DataFrame(csim_table,
                                                                         index=self.clustsim_ref['p'],
                                                                         columns=self.clustsim_ref['a'])

    @property
    def avail_tables(self):
        tables = sorted(self.clustsim_table.keys())
        return {i: tbl for i, tbl in enumerate(tables)}

    def get_clusterThr(self, table_idx, thr_p, thr_a):
        print('[get_clusterThr] p={}, a={}'.format(thr_p, thr_a))
        ref_a = self.clustsim_ref['a']
        xp = self.clustsim_ref['p']
        csim_table = self.clustsim_table[self.avail_tables[table_idx]]
        if thr_a not in ref_a:
            raise Exception('Choose a value from: {}'.format(ref_a))
        else:
            yp = csim_table[thr_a]
            if not np.all(np.diff(xp) > 0):
                xp = xp[::-1]
                yp = yp[::-1]
            cluster_size = np.interp(thr_p, xp, yp)
        return np.ceil(cluster_size).astype(int)

    def print_avail_brick(self):
        for i, test_params in [(k, v) for k, v in self.brick_statsymbol.items() if v != None]:
            print ("[print_avail_brick] i={}, label={}".format(i, self.brick_label[i]))

    def threshold_overlay(self, target_data, brick_idx):

        # img_data = self.nii.get_data()
        estimator = DBSCAN(eps=np.sqrt(self.NN), min_samples=1)

        test_params = self.brick_statsymbol[brick_idx]
        print("[threshold_overlay] brick_idx={}, param=={}".format(brick_idx, test_params))

        if test_params == None:
            raise Exception("[[[ERROR]]] [threshold_overlay] param is None!!")
        else:
            test_type, params = test_params
            idx_count = 0

            if test_type == 'Ftest':
                table_type = 'NN{}-2sided'.format(self.NN)
            elif test_type in ['Ttest', 'Zscore']:
                table_type = 'NN{}-bisided'.format(self.NN)
            else:
                raise Exception("[[[ERROR]]] [threshold_overlay] Not supported Stats, contact developer.")

            print("[threshold_overlay] table_type={}".format(table_type))

            if table_type not in self.avail_tables.values():
                raise Exception("[[[ERROR]]] [threshold_overlay]There is no ClusterSim Table for {}".format(table_type))
            table_idx = [j for j, ttype in self.avail_tables.items() if ttype == table_type][0]
            cluster_size = self.get_clusterThr(table_idx, self.thr_p, self.thr_a)
            print("[threshold_overlay] cluster_size={}".format(cluster_size))

            output_data = np.zeros(target_data.shape)

            if test_type in ['Ttest', 'Zscore']:
                if test_type == 'Ttest':
                    low_t = stats.t.ppf(self.thr_p / 2, params)
                    hgh_t = stats.t.ppf(1 - (self.thr_p / 2), params)
                elif test_type == 'Zscore':
                    low_t = stats.norm.ppf(self.thr_p / 2)
                    hgh_t = stats.norm.ppf(1 - (self.thr_p / 2))
                else:
                    raise Exception("[[[ERROR]]] [threshold_overlay] Check the test_type={}".format(test_type))

                low_X = np.transpose(np.nonzero(target_data <= low_t))
                hgh_X = np.transpose(np.nonzero(target_data >= hgh_t))

                survived_clusters_low = []
                survived_clusters_hgh = []

                if len(low_X) != 0:
                    low_clusters = estimator.fit(low_X).labels_
                    for c_idx in [c_idx for c_idx in set(low_clusters) if c_idx != -1]:
                        if len(low_clusters[low_clusters == c_idx]) >= cluster_size:
                            survived_clusters_low.append(c_idx)

                if len(hgh_X) != 0:
                    hgh_clusters = estimator.fit(hgh_X).labels_
                    for c_idx in [c_idx for c_idx in set(hgh_clusters) if c_idx != -1]:
                        if len(hgh_clusters[hgh_clusters == c_idx]) >= cluster_size:
                            survived_clusters_hgh.append(c_idx)

                if len(survived_clusters_low) != 0:
                    for c_idx in survived_clusters_low:
                        idx_count += 1
                        for x, y, z in low_X[np.where(low_clusters == c_idx)]:
                            output_data[x, y, z] = target_data[x, y, z]
                            # output_data[x, y, z] = idx_count
                if len(survived_clusters_hgh) != 0:
                    for c_idx in survived_clusters_hgh:
                        idx_count += 1
                        for x, y, z in hgh_X[np.where(hgh_clusters == c_idx)]:
                            output_data[x, y, z] = target_data[x, y, z]
                            # output_data[x, y, z] = idx_count

            elif test_type == 'Ftest':
                hgh_t = stats.f.ppf(1 - self.thr_p, *params)
                hgh_X = np.transpose(np.nonzero(target_data >= hgh_t))

                survived_clusters_hgh = []

                if len(hgh_X) != 0:
                    hgh_clusters = estimator.fit(hgh_X).labels_
                    for c_idx in [c_idx for c_idx in set(hgh_clusters) if c_idx != -1]:
                        if len(hgh_clusters[hgh_clusters == c_idx]) >= cluster_size:
                            survived_clusters_hgh.append(c_idx)

                if len(survived_clusters_hgh) != 0:
                    for c_idx in survived_clusters_hgh:
                        idx_count += 1
                        for x, y, z in hgh_X[np.where(hgh_clusters == c_idx)]:
                            output_data[x, y, z] = target_data[x, y, z]
                            # output_data[x, y, z] = idx_count

            output_data[output_data == 0] = np.nan

            return output_data

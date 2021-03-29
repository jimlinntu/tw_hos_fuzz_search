from pathlib import Path
import csv
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

import numpy as np
from pprint import pprint
import argparse

DEFAULT_DIR = Path(__file__).parent
DATA_DIR = DEFAULT_DIR / "data"

HOSPBSC_PATH = DATA_DIR / "hospbsc.txt"
EXTEND_HOSPITALS_PATH = DATA_DIR / "extend_hospitals.txt"
REGION_PATH = DATA_DIR / "regions.txt"
DESIGNATED_TYPES_PATH = DATA_DIR / "designated_types.txt"
HOSTP_CODE_PATH = DATA_DIR / "hosp_code.txt"
CITIES_PATH = DATA_DIR / "cities.txt"

class SearchResultSpec():
    def __init__(self, region_id, region, hos_id, hos_name, address, confidence, type_name, designated_type):
        assert isinstance(region_id, int)
        assert isinstance(region, str)
        assert isinstance(hos_id, str)
        assert isinstance(hos_name, str)
        assert isinstance(address, str)
        assert isinstance(type_name, str)
        assert isinstance(designated_type, str)

        self.region_id = region_id
        self.region = region
        self.hos_id = hos_id
        self.hos_name = hos_name
        self.address = address
        self.confidence = float(confidence)
        self.type_name = type_name # 型態別名稱
        self.designated_type = designated_type# 特約類別說明

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return dict(region_id=self.region_id,
                    region=self.region,
                    hos_id=self.hos_id,
                    hos_name=self.hos_name,
                    address=self.address,
                    confidence=self.confidence,
                    type_name=self.type_name,
                    designated_type=self.designated_type)

class SearchQuerySpec():
    def __init__(self, query, k, region, designated_types, address):
        assert isinstance(query, str)
        assert isinstance(k, int)
        assert isinstance(region, str)
        assert isinstance(designated_types, list)
        assert isinstance(address, str)
        self.query = query
        self.k = k
        self.region = region
        self.designated_types = designated_types
        self.address = address # where the patient lives

    @staticmethod
    def fromDict(query_dict):
        return SearchQuerySpec(query_dict["query"], query_dict["k"],
                               query_dict.get("region", ""), query_dict.get("designated_types", []),
                               query_dict.get("address", ""))

    @staticmethod
    def check_valid(query_dict):
        if not isinstance(query_dict, dict):
            return False
        if not ("query" in query_dict and "k" in query_dict):
            return False
        query = query_dict["query"]
        k = query_dict["k"]

        try:
            str(query)
            assert int(k) >= 1
        except:
            return False

        return True

class SearchEngine():
    def __init__(self, hospbsc_path=None, extend_hospitals_path=None, region_path=None,
                 designated_types_path=None, hosp_code_path=None, cities_path=None,
                 debug=True):
        self.debug = debug

        self.hospbsc_path = HOSPBSC_PATH
        if hospbsc_path is not None:
            assert isinstance(hospbsc_path, Path)
            self.hospbsc_path = hospbsc_path

        self.extend_hospitals_path = EXTEND_HOSPITALS_PATH
        if extend_hospitals_path is not None:
            assert isinstance(extend_hospitals_path, Path)
            self.extend_hospitals_path = extend_hospitals_path

        self.region_path = REGION_PATH
        if region_path is not None:
            assert isinstance(region_path, Path)
            self.region_path = region_path

        self.designated_types_path = DESIGNATED_TYPES_PATH
        if designated_types_path is not None:
            assert isinstance(designated_types_path, Path)
            self.designated_types_path = designated_type_path

        self.hosp_code_path = HOSTP_CODE_PATH
        if hosp_code_path is not None:
            assert isinstance(hosp_code_path, Path)
            self.hosp_code_path = hosp_code_path

        self.cities_path = CITIES_PATH
        if cities_path is not None:
            assert isinstance(cities_path, Path)
            self.cities_path = cities_path

        self.hosp_code_map = self._parse_hosp_code()
        self.designated_types_map, self.designated_types_set = self._parse_designated_types()
        self.region_ids, hos_ids, hos_names, self.addresses, self.type_names, self.designated_types_strs \
            = self._parse_hospbsc(self.hosp_code_map, self.designated_types_map, remove_empty_address=True,
                    add_extend_hospitals=True)
        self.region_map, self.region_set = self._parse_region()
        self.city2idx, self.idx2city, self.addr2idx = self._parse_cities()

        self.city_indices = [self.addr2idx(addr) for addr in self.addresses]

        self.regions = [self.region_map[idx] for idx in self.region_ids]
        self.type_scores = self._build_type_scores_by_rules(self.type_names)

        self.hos_ids = hos_ids
        self.hos_names = hos_names
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform([" ".join(hos_name) for hos_name in self.hos_names])
        self.keywords = self.vectorizer.get_feature_names()

        if debug:
            print("There are {} hospitals in Taiwan".format(len(hos_names)))
            print("There are {} unique hospital names in Taiwan".format(len(set(hos_names))))
            uniq_hos_names = set(hos_names)
            counter = { hos_name: 0 for hos_name in uniq_hos_names }

            dup_hos_names = []

            for hos_name in hos_names:
                if counter[hos_name] == 1:
                    dup_hos_names.append(hos_name)
                counter[hos_name] += 1

            print("There are {} duplicate hosptial names in Taiwan".format(len(dup_hos_names)))

            print("self.tfidf_matrix.shape: {}".format(self.tfidf_matrix.shape))
            print("self.tfidf_matrix length of the feature vector: {}".format(self.tfidf_matrix.shape[1]))

    def _parse_hospbsc(self, hosp_code_map, designated_types_map, remove_empty_address, add_extend_hospitals):
        assert isinstance(remove_empty_address, bool)
        assert isinstance(add_extend_hospitals, bool)

        assert isinstance(hosp_code_map, dict)
        assert isinstance(designated_types_map, dict)

        region_ids, hos_ids, hos_names, addresses, type_names, designated_types_strs = \
            [], [], [], [], [], []
        with self.hospbsc_path.open(encoding="utf-8", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None) # ignore the header
            for row in reader:
                region_id, hos_id, hos_name, addr, designated_type_id, type_id, category = \
                    int(row[0].strip()), row[1].strip(),\
                    row[2].strip(), row[3].strip(), row[6].strip(), row[7].strip(), row[8].strip()

                if remove_empty_address and len(addr) == 0:
                    continue

                region_ids.append(region_id)
                hos_ids.append(hos_id)
                hos_names.append(hos_name)
                addresses.append(addr)
                type_names.append(hosp_code_map[(type_id, category)])

                if designated_type_id not in designated_types_map:
                    if self.debug:
                        print("衛福部的朋友, {}'s 特約類別 {} 在對照表沒出現過! 修一下好嗎".format(hos_name, designated_type_id))
                    # 用 不詳 來代替
                    designated_types_strs.append(designated_types_map["X"])
                else:
                    designated_types_strs.append(designated_types_map[designated_type_id])

        if add_extend_hospitals:
            with self.extend_hospitals_path.open(encoding="utf-8", newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=",")
                next(reader, None)
                for row in reader:
                    region_id, hos_id, hos_name, addr, designated_type_id, type_id, category = \
                        int(row[0].strip()), row[1].strip(),\
                        row[2].strip(), row[3].strip(), row[6].strip(), row[7].strip(), row[8].strip()

                    if remove_empty_address and len(addr) == 0:
                        continue

                    region_ids.append(region_id)
                    hos_ids.append(hos_id)
                    hos_names.append(hos_name)
                    addresses.append(addr)
                    type_names.append("") # we don't know the extend hospital's type_name!

                    assert designated_type_id in designated_types_map, "自己加的 {} 自己負責!".format(self.extend_hospitals_path)
                    designated_types_strs.append(designated_types_map[designated_type_id])

        return region_ids, hos_ids, hos_names, addresses, type_names, designated_types_strs

    def _parse_region(self):
        region_map = dict()
        region_set = set()
        with self.region_path.open(encoding="utf-8", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for row in reader:
                region_id = int(row[0])
                region = row[1]

                region_map[region_id] = region
                region_set.add(region)
        return region_map, region_set

    def _parse_hosp_code(self):
        hosp_code_map = dict()
        with self.hosp_code_path.open(encoding="utf-8", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for row in reader:
                type_id, category = row[0], row[2]
                type_name = row[1]
                hosp_code_map[(type_id, category)] = type_name
        return hosp_code_map

    def _parse_designated_types(self):
        designated_types_map = dict()
        designated_types_set = set()

        with self.designated_types_path.open(encoding="utf-8", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for row in reader:
                type_id, type_str = row[0], row[1]
                designated_types_map[type_id] = type_str
                designated_types_set.add(type_str)

        return designated_types_map, designated_types_set

    def _parse_cities(self):
        city2idx = dict()
        cities = []
        idx2city = ["UNKNOWN_CITY"]

        with self.cities_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0: continue
                city_and_aliases = line.split(",")
                assert len(city_and_aliases) >= 1

                cities.extend(city_and_aliases)

                for c in city_and_aliases:
                    city2idx[c] = len(idx2city)

                idx2city.append(city_and_aliases[0])

        # Convert addr to an index of city
        def addr2idx(addr):
            assert isinstance(addr, str)
            addr = addr.replace(" ", "")
            for c in cities:
                if c in addr:
                    return city2idx[c]
            return 0

        return city2idx, idx2city, addr2idx

    def _build_type_scores_by_rules(self, type_names):
        '''
            根據 型態別 給予加分
            Ex. 綜合醫院 should have higher scores while 藥局 will have lower score
        '''
        type_scores = []
        for type_name in type_names:
            score = 0.
            if "綜合" in type_name:
                score += 0.4
            if "醫院" in type_name:
                score += 0.3
            if "診所" in type_name:
                score += 0.1
            type_scores.append(score)
        return type_scores

    def query2matrix(self, query):
        assert isinstance(query, str)
        query_matrix = self.vectorizer.transform([" ".join(query)])
        return query_matrix

    def compute_kernel_matrix(self, query_matrix):
        assert query_matrix.shape == (1, len(self.keywords))
        kernel = cosine_similarity(query_matrix, self.tfidf_matrix)
        assert kernel.shape == (1, len(self.hos_names))
        return kernel.ravel()

    def compute_contain_substr(self, query):
        assert isinstance(query, str)
        scores = np.array([int(query in hos_name) for hos_name in self.hos_names], dtype=np.int32)
        return scores

    def weighted_scores(self, scores, substr_scores):
        return 0.5 * scores + 0.3 * np.array(self.type_scores) + 0.2 * substr_scores

    def preprocess_query(self, query):
        query = re.sub("[0-9]", "", query)
        return query

    def search(self, query_spec, preprocess=True):
        assert isinstance(query_spec, SearchQuerySpec)
        query_hos_name = query_spec.query
        if preprocess:
            query_hos_name = self.preprocess_query(query_hos_name)
        if len(query_hos_name) == 0:
            return []

        k = query_spec.k
        region = query_spec.region
        designated_types = query_spec.designated_types
        effective_designated_types_set = set([d_type for d_type in designated_types if d_type in self.designated_types_set])

        assert isinstance(query_hos_name, str)
        query_matrix = self.query2matrix(query_hos_name)
        scores = self.compute_kernel_matrix(query_matrix)
        substr_scores = self.compute_contain_substr(query_hos_name)

        scores = self.weighted_scores(scores, substr_scores)
        # Generate the mask for 分區別
        scores_mask = np.ones((len(scores), ), dtype=np.int32)
        if region in self.region_set:
            mask_indices = [i for i, r in enumerate(self.regions) if region != r]
            scores_mask[mask_indices] = 0 # ignore these scores

        # Generate the mask for 特約類別
        mask_indices = [i for i, d_type in enumerate(self.designated_types_strs) if d_type not in effective_designated_types_set]
        scores_mask[mask_indices] = 0

        # Apply the mask
        scores = scores * scores_mask

        retrieved_indices = []

        if len(query_spec.address) == 0:
            if k == 1:
                idx = np.argmax(scores)
                retrieved_indices.append(idx)
            else:
                # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
                # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
                # argpartition will gaurantee the first kth element (if k > 0) in sorted (ascending) order
                # and the last kth element in sorted (ascending) order

                # O(n) (introselect) + O(k log k) (sorting)
                indices = np.argpartition(scores, -k)
                topk_indices_unsorted = indices[-k:]
                topk_indices_sorted = topk_indices_unsorted[np.argsort(scores[topk_indices_unsorted])]
                topk_indices_sorted = topk_indices_sorted[::-1]

                retrieved_indices = topk_indices_sorted
        else:
            # sort (score, (the city of address == this hospital's city))
            # the second element of the tuple will only effective if scores tie

            # O(n log n)
            sorted_indices = sorted(range(len(self.hos_names)),
                    key=lambda i: (scores[i], int(self.city_indices[i] == self.addr2idx(query_spec.address))),
                    reverse=True)
            retrieved_indices = sorted_indices[:k]

        results = [SearchResultSpec(
                    region_id=self.region_ids[i], region=self.regions[i],
                    hos_id=self.hos_ids[i], hos_name=self.hos_names[i],
                    address=self.addresses[i], confidence=scores[i], type_name=self.type_names[i],
                    designated_type=self.designated_types_strs[i])
                    for i in retrieved_indices]
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="hospital query string")
    parser.add_argument("k", type=int, help="topk's k")
    parser.add_argument("--region", type=str, default="", help="業務組名稱 filter. See tw_hos_fuzz_search/data/regions.txt for supported region values")
    parser.add_argument("--designated_types", type=str, nargs="*", help="特約類別說明的 filter. See tw_hos_fuzz_search/data/designated_types.txt",
                        default=["醫學中心", "區域醫院", "地區醫院", "診所"])
    parser.add_argument("--address", default="", help="使用者(病患)的地址")

    args = parser.parse_args()
    se = SearchEngine()
    results = se.search(SearchQuerySpec(args.query, args.k, args.region,
                                        args.designated_types, args.address))
    pprint(results)

if __name__ == "__main__":
    main()

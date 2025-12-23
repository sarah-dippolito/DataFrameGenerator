# imports
import pandas as pd
import numpy as np
import string

'''
Class that generates dataframes based on chosen column types and parameter values 
'''

# start class and introduce random state (if applicable)
class DataFrameGenerator:
    def __init__(self, nrows, random_state=None):
        self.nrows = nrows
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.df = pd.DataFrame(index=range(nrows))
        
    # apply certain percentage of missing values for practice data
    def _apply_missing(self, series, percent):
        if percent <= 0:
            return series
        mask = np.random.rand(len(series)) < percent
        series = series.copy()
        series[mask] = np.nan
        return series
    
    # add columns
    def add_column(self, name, part):
        col_type = part[0]
        meta = part[-1] if isinstance(part[-1], dict) else {}
        missing_pct = meta.get("missing", 0.0)

        # integer (parameters: min, max)
        if col_type == "int":
            min, max = part[1], part[2]
            data = np.random.randint(min, max + 1, self.nrows)
            
        # weighted integer (parameters: values, probability-per-integer)
        elif col_type == "int_prob":
            values, probs = part[1], part[2]
            data = np.random.choice(values, self.nrows, p=probs)


        # uniform/float/decimals (parameters: min, max)
        elif col_type in ("uniform", "float", "decimal"):
            min, max = part[1], part[2]
            data = np.random.uniform(min, max, self.nrows)

        # boolean (parameters: none)
        elif col_type == "bool":
            data = np.random.choice([True, False], self.nrows)

        # bernoulli (parameter: probability)
        elif col_type == "bernoulli":
            p = part[1]
            data = np.random.rand(self.nrows) < p
        
        # percentages (parameters: alpha, beta)
        elif col_type == "percent":
            alpha, beta_param = part[1], part[2]
            data = np.random.beta(alpha, beta_param, self.nrows)


        # categorical (parameter: categories)
        elif col_type in ("cat", "choice"):
            categories = part[1]
            data = np.random.choice(categories, self.nrows)
        
        # weighted categorical (parameters: categories, probabilities-per-choice)
        elif col_type == "cat_prob":
            categories, probs = part[1], part[2]
            data = np.random.choice(categories, self.nrows, p=probs)

        # string (parameter: length)
        elif col_type == "string":
            length = part[1]

            def rand_str():
                return "".join(np.random.choice(list(string.ascii_letters), length))

            data = [rand_str() for _ in range(self.nrows)]

        # id (parameter: start_value)
        elif col_type == "id":
            start = part[1] if len(part) > 1 else 1
            data = np.arange(start, start + self.nrows)
            
        # alphanumeric id (parameters: length, unique [optional])
        elif col_type == "alphanumeric_id":
            length = part[1]
            unique = part[2] if len(part) > 2 else True
            chars = list(string.ascii_letters + string.digits)
            def rand_id():
                return "".join(np.random.choice(chars, size=length))
            if unique:
                ids = set()
                while len(ids) < self.nrows:
                    ids.add(rand_id())
                data = list(ids)
            else:
                data = [rand_id() for _ in range(self.nrows)]
            
        # id with prefix (parameters: prefix, number_integers, unique [optional])
        elif col_type == "prefix_id":
            prefix = part[1]
            n_ints = part[2]
            unique = part[3] if len(part) > 3 else True

            if unique:
                nums = np.arange(1, self.nrows + 1)
                data = [f"{prefix}{str(n).zfill(n_ints)}" for n in nums]
            else:
                max_val = 10**n_ints - 1
                nums = np.random.randint(1, max_val + 1, self.nrows)
                data = [f"{prefix}{str(n).zfill(n_ints)}" for n in nums]


        # date (parameters: start_date, end_date)
        elif col_type == "date":
            start, end = pd.to_datetime(part[1]), pd.to_datetime(part[2])
            data = pd.to_datetime(
                np.random.randint(
                    start.value // 10**9,
                    end.value // 10**9,
                    self.nrows
                ),
                unit="s"
            ).normalize()

        # datetime (parameters: start_date, end_date)
        elif col_type == "datetime":
            start, end = pd.to_datetime(part[1]), pd.to_datetime(part[2])
            data = pd.to_datetime(
                np.random.randint(
                    start.value // 10**9,
                    end.value // 10**9,
                    self.nrows
                ),
                unit="s"
            )

        # normal distribution (parameters: mean, standard_deviation)
        elif col_type == "normal":
            mean, sd = part[1], part[2]
            data = np.random.normal(mean, sd, self.nrows)

        # lognormal distribution (parameters: mean, sigma)
        elif col_type == "lognormal":
            mean, sigma = part[1], part[2]
            data = np.random.lognormal(mean, sigma, self.nrows)
            
        # normal distribution with min and max (parameters: mean, standard_deviation, min, max)
        elif col_type == "normal_clip":
            mean, sd, min, max = part[1], part[2], part[3], part[4]
            data = np.random.normal(mean, sd, self.nrows)
            data = np.clip(data, min, max)

        # poisson distribution (parameter: lam)
        elif col_type == "poisson":
            lam = part[1]
            data = np.random.poisson(lam, self.nrows)
            
        # beta distribution (parameter: alpha, beta, max)
        elif col_type == "beta":
            alpha, beta_param, max = part[1], part[2], part[3]
            data = np.random.beta(alpha, beta_param, self.nrows) * max

        # correlated column - linear (parameters: base_column, slope, standard_deviation)
        elif col_type == "correlated":
            base_col, slope, noise_sd = part[1], part[2], part[3]
            base = self.df[base_col]
            noise = np.random.normal(0, noise_sd, self.nrows)
            data = slope * base + noise

        else:
            raise ValueError(f"Column type is unknown: {col_type}") #no such column type found

        # apply missing to specified columns
        self.df[name] = self._apply_missing(pd.Series(data), missing_pct)
        return self  

    # build dataframe
    def build(self):
        return self.df.copy()
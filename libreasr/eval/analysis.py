import pickle

import torch
import torchaudio

from IPython.display import Audio, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libreasr.lib.utils import sanitize_str


class Analysis:
    def __init__(
        self, learn, num_samples, pcent, builder_valid, results=None, **kwargs
    ):
        self.learn = learn
        self.num_samples = num_samples
        self.pcent = pcent
        self.builder_valid = builder_valid
        self.results = results
        self.kwargs = kwargs

    def _evaluate(self):
        assert self.results is None, "Results already available"
        assert self.learn is not None
        assert self.num_samples is not None
        assert self.pcent is not None

        # grab evaluator
        eval_fn = self.learn.evaluator.evaluator.eval

        # perform evaluation
        kwargs = self.kwargs
        self.results = list(
            eval_fn(
                self.learn,
                num_samples=self.num_samples,
                train=False,
                save_best=False,
                pcent=self.pcent,
                **kwargs,
            )
        )

    def _get_results(self):
        if self.results is None:
            self._evaluate()

    def _sort(self, rev):
        self._get_results()
        results = self.results
        results = filter(lambda x: "text/ground_truth" in list(x.keys()), results)
        results = sorted(results, key=lambda x: x["metrics/wer"], reverse=rev)
        return results

    @staticmethod
    def load(file_name, builder_valid=None):
        loaded = pickle.load(open(file_name, "rb"))
        if isinstance(loaded, dict) and "kwargs" in loaded:
            results = loaded["results"]
            kwargs = loaded["kwargs"]
        else:
            results = loaded
            kwargs = {}
        return Analysis(None, None, None, builder_valid, results=results, **kwargs)

    def save(self, file_name, save_kwargs=True):
        assert self.results is not None
        if save_kwargs:
            to_save = {
                "results": self.results,
                "kwargs": self.kwargs,
            }
        else:
            to_save = self.results
        pickle.dump(to_save, open(file_name, "wb"))

    def cer(self):
        self._get_results()
        return self.results[-1]["metrics/mean_cer"]

    def wer(self):
        self._get_results()
        return self.results[-1]["metrics/mean_wer"]

    def best(self):
        self._get_results()
        return self._sort(rev=False)

    def summary(self):
        self._get_results()

        ns = len(self.results) - 1
        first_g = self.results[0]["text/ground_truth"]
        first_p = self.results[0]["text/prediction"]
        print(f"\nSummary:\n")
        print("CER:", self.cer())
        print("WER:", self.wer())
        print("Evaluated on # samples:", ns)
        print("First ground truth:", first_g)
        print("First prediction:  ", first_p)
        print("kwargs:", self.kwargs)

    def worst(self, n=5):
        self._get_results()

        print(f"\nShowing {n} examples with the worst WERs:\n")
        results = self.results
        results = self._sort(rev=True)[:n]
        df = self.builder_valid.df
        for res in results:

            def filter_fn(row):
                s1 = res["text/ground_truth"]
                s2 = sanitize_str(row.label.decode("utf-8"))
                in_ok = s1 in s2
                l_ok = abs(len(s1) - len(s2)) < 10
                return in_ok and l_ok

            rows = df[df.apply(filter_fn, axis=1)]
            row = rows
            if len(row) == 1:
                row = row.iloc[0]
                file = row.file
                sr = int(row.sr)
                xstart = int(row.xstart / 1000.0 * sr)
                xlen = int(row.xlen / 1000.0 * sr)
                data, sr = torchaudio.load(file, frame_offset=xstart, num_frames=xlen)
                print("PRED:", res["text/prediction"])
                print("TRUE:", res["text/ground_truth"])
                print(f"WER:  {res['metrics/wer'] * 100.:.2f}%")
                display(Audio(data.numpy(), rate=sr))
            elif len(row) > 1:
                print(f"Warning: {len(row)} rows for one item...")
            else:
                print("Warning: No row for one item...")

    def plot(self, save=False, save_name="analysis-plot.png", save_dpi=200):
        self._get_results()

        print("\nPlotting WER by label length:\n")

        # prepare for plotting
        results = self.results
        ylens = map(lambda x: x.get("text/ground_truth"), results)
        ylens = filter(lambda x: x is not None and len(x) != 0, ylens)
        ylens = map(lambda x: len(x), ylens)
        ylens = np.array(list(ylens))
        wers = map(lambda x: x.get("metrics/wer", None), results)
        wers = filter(lambda x: x is not None, wers)
        wers = np.array(list(wers))
        print(ylens[:3], wers[:3], len(ylens), len(wers))

        # create df
        stack = np.stack([ylens, wers]).T
        df = pd.DataFrame(stack, columns=["ylen", "wer"])
        print(df.head())

        # stats
        mean_wer = df.groupby("ylen").mean()["wer"].values
        std_wer = df.groupby("ylen").std()["wer"].values
        border_upper = mean_wer + std_wer / 2
        border_lower = mean_wer - std_wer / 2
        count = df.groupby("ylen").count()["wer"].values

        # actually plot
        fig = plt.figure()
        plot_wer = fig.add_subplot(111)
        plot_count = plot_wer.twinx()
        x = np.arange(len(mean_wer))

        # wer
        plot_wer.plot(mean_wer)
        plot_wer.set_ylim(0.0, 1.2)
        plot_wer.set_ylabel("WER")
        plot_wer.set_xlabel("Label Length")

        # std
        plot_wer.fill_between(x, border_upper, border_lower, alpha=0.2)

        # count
        plot_count.bar(x, count, alpha=0.1)
        plot_count.set_ylabel("Number of Examples")
        plot_count.set_yscale("log")

        # show & save plot
        plt.show()
        if save:
            plt.savefig(save_name, dpi=save_dpi)

    def all(self):
        """
        Do a full analysis
        (evaluate + worst + plot)
        """
        self.summary()
        self.worst()
        self.plot()

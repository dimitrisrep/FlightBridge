from pathlib import Path
import hashlib
import platform

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from prepare_data import (BASE_DIR, DATA_FILE, CATEGORICAL, NUMERIC, TARGETS,
                          load_data, save_prepared_data)

matplotlib.use("Agg")  #gia backend grafimata se png#
import matplotlib.pyplot as plt #sxediazei ta grafimata kai ta kanei save#
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier #k-NN eksetazei same old cases#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os

RANDOM_STATE = 42
MODEL_FILE = BASE_DIR / "model.pkl"
FIGURES_DIR = BASE_DIR / "figures"

# Το load_data βρίσκεται στο prepare_data.py: ελέγχει pap_request + solution
# για κάθε δικαίωμα χωριστά. Οι στήλες αποτελέσματος δεν μπαίνουν στο X.


def build_preprocessor():
    
    return ColumnTransformer(
        transformers=[
            ("categorical_columns", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), #onoma|ergalio|stili. metatrepei tis katigorikes stiles se arithmous.s
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),("scale", StandardScaler()), ]), NUMERIC),
        ] #an leipei timi, pairnw diameso# #metatropi arithmwn se koini klimaka girw apo 0#
    )
    #to preprocessor metatrepei lekseis se arithmous, symplirwnei kena delay_minutes kai kanei scaling stous arithmous#

def get_models(): #leksiko me 3 modela pou tha sigkrithoun#
    return {
        "Decision Tree": DecisionTreeClassifier(  
            max_depth=4, 
            min_samples_leaf=2, 
            random_state=RANDOM_STATE), #kathe dento max 4 epipeda#
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=5, 
            min_samples_leaf=2, #200 dentra apofasis, provlepseis sindiazontai, kathe fylo 2 periptwseis#
            random_state=RANDOM_STATE),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5), #gia nea periptwsi eksetazontai oi 5 pio kontines palies#
    }


def evaluate_models(features: pd.DataFrame, answer: pd.Series): #fearures pinakas me stoixeia eisodou, answer einai stili me tis swstes apantisis yes/no gia ton sygkekrimeno stoxo#

    has_one_class = answer.nunique() < 2 #metraei poses diaforetikes times yparxoun
    if has_one_class or answer.value_counts().min() < 2:
        raise ValueError("Για αξιολόγηση χρειάζονται τουλάχιστον δύο YES και δύο NO ανά στόχο.")
    n_splits = min(5, int(answer.value_counts().min()))
    crossvalidation = StratifiedKFold(n_splits=n_splits, shuffle=True, #CrossValidation, shuffle anakatevei grammes, random state gia repeat idio anakatema#
                         random_state=RANDOM_STATE)
    folds = list(crossvalidation.split(features, answer))

    rows, predictions = [], {} #dimiourgei mia lista me rows kai ena leksiko predictions#
    for name, classifier in get_models().items(): #name einai onoma px random forest, clf einai to classifier (diladi to idio to randrom forest classifier)#
        pipe = Pipeline([("preparation", build_preprocessor()), ("classifier", classifier)]) #proetoimase ta dedomena kai meta dwse ta dedomena sto classifier#
        answer_pred = cross_val_predict(pipe, features, answer, cv=folds) #X= stoixeia pou xrisimopoiw gia provlepsi, y= swstes apantiseis, cv= cross validation#
        predictions[name] = answer_pred
        rows.append({ #vazw stin lista rows ena dictionary me metrics gia kathe model#
            "Μοντέλο": name,
            "Accuracy": accuracy_score(answer, answer_pred), #pososto swstwn provlepsewn apo to model#
            "Precision": precision_score(answer, answer_pred, average="weighted", #otan to model provlepei mia klasi, poso syxna exei dikio#
                                         zero_division=0),
            "Recall": recall_score(answer, answer_pred, average="weighted", # poses pragmatikes periptwseis mias klasis entwpise#
                                   zero_division=0),#to onomasan weighted giati kathe klasi pairnei diaforetiko varos analoga me to poses pragmatikes periptwseis#
            "F1-score": f1_score(answer, answer_pred, average="weighted", #precision kai recall se ena score, epilegei to kalutero model#
                                 zero_division=0), 
        })                                 #exw orisei panw pws "Modelo":name#
    results = pd.DataFrame(rows).set_index("Μοντέλο")
    results.attrs.update({"has_one_class": has_one_class, "n_splits": n_splits,
                          "predictions": predictions})
    return results
###Kathe periptwsi xrisimopoieitai polles fores ws ekpaideftiko paradeigma kai mia fora ws agnwsti periptwsi eksetasis. Etsi h aksiologisei einai pio
# aksiopisti apo to na dokimazame to modelo panw stis idies grammes pou molis emathe###

def make_figures(target: str, results: pd.DataFrame, features: pd.DataFrame,#den ekpedevei to modelo, dimiourgei images. giafto epistrefei None, den exei return#
                 answer: pd.Series, best_name: str, best_pipe: Pipeline) -> None:
    
    FIGURES_DIR.mkdir(exist_ok=True) #dimiourgei ton fakelo figures#
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150}) #megeth font size kai dpi gia ta grafimata#
    tag = target.replace("_allowed", "") #svinei to _allowed gia na exei ligotero gramma sto onoma tou arxeiou#
    # Μην παραμείνει παλιό γράφημα αν ο νέος νικητής δεν έχει feature_importances_.
    for name in ["model_comparison", "confusion_matrix", "feature_importance", "decision_tree"]:
        (FIGURES_DIR / f"{name}_{tag}.png").unlink(missing_ok=True)

    ax = results.plot.bar(figsize=(8, 4.5), rot=0, width=0.78, #dimiourgei bar char me metrics twn 3 modelwn, ax= axonas#
                          color=["#4269d0", "#efb118", "#ff725c", "#6cc5b0"],
                          edgecolor="white") 
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Τιμή μετρικής")
    ax.set_title(f"Σύγκριση αλγορίθμων — {TARGETS[target]}")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"model_comparison_{tag}.png", bbox_inches="tight")
    plt.close()  #kleinei to grafima apo tin mnimi#

    if results.attrs["has_one_class"]: #an yparxei mono mia klasi, den exei noima na dimiourgisoume confusion matrix h dentro apofasis#
        return  # confusion matrix / δέντρο δεν έχουν νόημα με μία μόνο κλάση,

    # 2. Confusion matrix του καλύτερου μοντέλου
    answer_pred = results.attrs["predictions"][best_name] #pare tis provlepseis tou kaluterou modelou apo to results.attrs["predictions"]
    labels = sorted(answer.unique()) #vriskei tis diaforetikes klasis tou y kai tis taksinomei alfavitika
    cm = confusion_matrix(answer, answer_pred, labels=labels)#pare ti provlepsi tou kaluterou modelou kai to swsto answer kai dimiourgei confusion matrix, h malakia me ta TP FP TN kai FN
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Πρόβλεψη")
    ax.set_ylabel("Πραγματική τιμή")
    ax.set_title(f"Confusion matrix — {best_name}\n({TARGETS[target]})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{tag}.png", bbox_inches="tight")
    plt.close()

    # 3. Σημαντικότητα χαρακτηριστικών (αν υποστηρίζεται)
    classifier = best_pipe.named_steps["classifier"] #To named_steps einai leksiko me ta steps tou pipeline, kai pairnw to classifier tou kaluterou modelou

    if hasattr(classifier, "feature_importances_"): # vlepei an to modelo ypostirizei feature_importances_ (diladi an einai dentro apofasis h random forest)

        names = best_pipe.named_steps["preparation"].get_feature_names_out()#pare apo to pipeline tou kalyterou modelou to vhma preparation(proepeksergasia dedomenwn)
        #kai  get... dwse ta onomata twn features pou exoun metatrapei se arithmous apo to preprocessor

        names = [n.split("__")[-1] for n in names] #afairei to prefix cat__ kai num__ apo panw. perase apo kathe onoma ths listas names. Sto "__" tou lew na diaxwrisei
        #to keimeno ekei pou exei ta 2 underscore. -1 einai to teleftaio, afto tou leei na kratisei#

        importance = pd.Series(classifier.feature_importances_, names).sort_values()#Series= monodiastati stili dedomenwn. edw lew na parei ta feature importances tou classifier
        #mazi me to onoma, kai meta sortarei ta values tou Series me vasi tis times twn feature importances#

        importance = importance[importance > 0].tail(15)
        ax = importance.plot.barh(figsize=(8, max(4.5, len(importance) * 0.35)), color="#4269d0",
                           edgecolor="white")
        ax.set_title(f"Σημαντικότητα χαρακτηριστικών — {TARGETS[target]}")
        ax.set_xlabel("Σημαντικότητα")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"feature_importance_{tag}.png", bbox_inches="tight")
        plt.close()  #to plt. einai pyplot apo to matplotlib)

    # 4. Οπτικοποίηση δέντρου απόφασης
    decision_tree_pipe = Pipeline([("preparation", build_preprocessor()),
                          ("classifier", DecisionTreeClassifier(
                              max_depth=4, min_samples_leaf=2,
                              random_state=RANDOM_STATE))]) #dimiourgei ena pipeline me to preprocessor kai to classifier pou einai dentro apofasis, aneksartita
    #an to kalytero model einai allo, to kanw fit me to dentro apofasis gia na to emfanisw#

    decision_tree_pipe.fit(features, answer)#.ft simainei ekpaidevse to pipeline xrisimopoiontas dedomena X kai swstes apantiseis y. to dentro apofasis tha emfanistei sto grafima.
    #Den mporw na sxediasw ena dentro prin ekpaideftei#

    names = decision_tree_pipe.named_steps["preparation"].get_feature_names_out()
    names = [n.split("__")[-1] for n in names]
    plt.figure(figsize=(13, 7))
    plot_tree(
        decision_tree_pipe.named_steps["classifier"], 
        feature_names=names,
        class_names=list(decision_tree_pipe.classes_), filled=True, rounded=True,
        fontsize=14)
    plt.title(f"Δέντρο απόφασης — {TARGETS[target]}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"decision_tree_{tag}.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_data() #anoigei kai kathatizei to csv arxeio kai epistrefei ena clean dataframe#
    save_prepared_data(df)
    known_airlines = sorted(
        df["airline"].dropna().unique().tolist() #pairnei stili, afairei kenes times,krata kathe airline mia fora, to kanei lista#
    )
#to .unique ta kanei numpy array, giafto ta kanw lista me to .tolist meta#

    airline_type_map = (
        df.dropna(subset=["airline", "airline_type"]) #afairei tis grammes pou exoun NaN sto airline h airline_type, giafto to subset#
        .drop_duplicates(subset=["airline"]) #afairei tis diples grammes pou exoun to idio airline#
        .set_index("airline")["airline_type"]
        .to_dict()
    )

    bundle = {
        "schema_version": 2,
        "dataset_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "data_audit": df.attrs["audit"],
        "features": {
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,},
        "known_airlines": known_airlines,
        "airline_type_map": airline_type_map,
        "targets": {},
       
    }

    for target, label in TARGETS.items(): #epanalipsi 2 foeres, mia gia refund, mia gia rebooking#
        print("=" * 62)
        print(f"ΣΤΟΧΟΣ: {target}  ({label})")
        print("=" * 62)
        sub = df.dropna(subset=[target]) #afairei tis grammes pou exoun NaN sto target. to subset= to target pou vlepei twra, eite einai refund eite rebooking (allazei)
        #analoga to target, svinei grammes pou afto exei NaN#
        
        X, y = sub[CATEGORICAL + NUMERIC], sub[target]#dimiourgei taftoxrona ta features X kai to target y. X= stili me ta features, y= stili me to target#
        print(f"{len(sub)} περιπτώσεις | Κατανομή: " #kanei print poses periptwseis yparxoun kai posa yes/no exei o stoxos#
              f"{y.value_counts().to_dict()}")

        results = evaluate_models(X, y)
        print(f"\nΑποτελέσματα ({results.attrs['n_splits']}-fold "
              f"stratified cross-validation):")
        print(results.round(3).to_string()) #stroggylopoiei ta metrics se 3 dekadika, to kanei string kai ta kanei print#

        best_name = results["F1-score"].idxmax() #pare to onoma tou modelou me to megalytero F1-score, to idxmax() epistrefei to index tou megalyterou#

        #dimiourgei neo pipeline me ton kalytero classifier#
        best_pipe = Pipeline([("preparation", build_preprocessor()),
                              ("classifier", get_models()[best_name])])
        best_pipe.fit(X, y)  # τελική εκπαίδευση σε όλο το dataset#
        print(f"\nΚαλύτερο μοντέλο: {best_name}")
        if not results.attrs["has_one_class"]:
            print(classification_report(
                y, results.attrs["predictions"][best_name], zero_division=0)) #to zero division leei pws na se kapoion ypologismo xriastei diairesei me to 0
            #na min faei error, alla na epistrefei 0
        print()

        bundle["targets"][target] = {
            "label": label,
            "pipeline": best_pipe,
            "model_name": best_name,
            "classes": sorted(y.unique()),
            "results": results,
            "training_count": len(sub),
            "class_counts": y.value_counts().to_dict(),
            "label_sources": df[target + "_source"].value_counts().to_dict(),
            "majority_baseline": y.value_counts(normalize=True).max(),
        }
        make_figures(target, results, X, y, best_name, best_pipe) # dimiourgei ta grafimata kai ta apothikevei sto fakelo figures/#

    joblib.dump(bundle, MODEL_FILE) # apothikevei to leksiko bundle me to model.pkl#
    print(f"Τα μοντέλα αποθηκεύτηκαν στο {MODEL_FILE}")
    print("Τα γραφήματα αποθηκεύτηκαν στον φάκελο figures/")


if __name__ == "__main__":  #gia na to travaw apo to command line kai na trexei to main()#
    main()

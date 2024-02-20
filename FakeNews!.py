'''
Created on Apr 15, 2020

@author: Nick
'''


import csv
import textblob
import matplotlib.pyplot as plt
import sklearn.neural_network
import sklearn.metrics
import syllables
import math
import statistics as stats
import tkinter
import joblib
import sys



def preprocess_data():
    """Preprocesses data for Machine Learning. 1 denotes Fake, 0 denotes True"""
    fake_news_list = []
    true_news_list = []
    
    try:
        with open("Fake.csv", "r",encoding="utf-8") as file:  #Attempts to open Fake News set for the ML
            y_list_fake = []
            reader = csv.reader(file)
            for row in reader:
                
                temp_list = []
                title = row[0]
                review = row[1]
                avg_polarity = polarity(title, review)
        
                avg_subjectivity = subjectivity(title, review)
                
                avg_formality = formality(title, review)
                
                avg_fkgl , avg_smog = readability_fkgl_smog(title, review)
                
                temp_list.extend([avg_polarity,avg_subjectivity,avg_formality,avg_fkgl,avg_smog])
    
                fake_news_list.append(temp_list)
                y_list_fake.append(1)
                
        with open("True.csv", "r",encoding="utf-8") as file: #Attempts to open the True News for ML
            reader = csv.reader(file)
            y_list_truth = []
            for row in reader:
                
                temp_list = []
                title = row[0]
                review = row[1]
                avg_polarity = polarity(title, review)
    
                avg_subjectivity = subjectivity(title, review)
                
                avg_formality = formality(title, review)
                
                avg_fkgl , avg_smog = readability_fkgl_smog(title, review)
                
                temp_list.extend([avg_polarity,avg_subjectivity,avg_formality,avg_fkgl,avg_smog])
                
                true_news_list.append(temp_list)
                y_list_truth.append(0)
                
        x_final_list = fake_news_list + true_news_list
        y_final_list = y_list_fake + y_list_truth            
        
        return x_final_list, y_final_list, fake_news_list, true_news_list
    except:
        print("Base Machine Learning files could not be found, please run the program again.")
        sys.exit()
            
    

def polarity(title,text):
    """Determines the polarity of the text."""
    blob = textblob.TextBlob(title)
    blob1 = textblob.TextBlob(text)
    polarity_title = blob.polarity
    polarity_text = blob1.polarity
    avg_polarity = (polarity_text + polarity_title) / 2
    
    return avg_polarity
    
        

def subjectivity(title,text):
    """Determines how subjective the text is."""
    blob = textblob.TextBlob(title)
    blob1 = textblob.TextBlob(text)
    subject_title = blob.subjectivity
    subject_text = blob1.subjectivity
    avg_subject = (subject_text + subject_title) / 2
    
    return avg_subject

def formality(title,text):
    """Determines the formality of the text given."""
    f_list = ["NN","NNP", "NNPS", "JJ", "JJR", "JJS","IN", "DT", "CC", "PDT", "TO","POS"]
    
    c_list = ["PRP", "PRP$", "RB","RBR","RBS","UH","VB","VBD","VBG","VBN","VBP","VBZ","WP","WP$","WRB"]
    
    formality_temp = 0
    formality_avg = 0
    formality_total  = 0
    f = 0
    c = 0
    
    blob = textblob.TextBlob(title.lower())
    words = blob.words
    for words,tag in blob.tags:
        if tag in f_list: #Checks to see the part-of-speech of each word.
            f += 1
        elif tag in c_list:
            c += 1
        
        else:
            pass
    
    try:    
        formality_avg = (50 * ( (f-c)/(f+c) + 1)) #Runs calculation, adds it to the running total.
    except:
        formality_avg = 1
            
    formality_total_title = formality_avg / (len(words) + 1)
    blob1 = textblob.TextBlob(text)
    words = blob1.words
    for words,tag in blob1.tags:

        if tag in f_list: #Checks to see the part-of-speech of each word.
            f += 1
        elif tag in c_list:
            c += 1
        
    try:    
        formality_avg = (50 * ( (f-c)/(f+c) + 1)) #Runs calculation, adds it to the running total.
        formality_total_text = formality_avg / len(words)
    except:
        formality_avg = 0
        formality_total_text = formality_avg / 1
            
            
    
    
    formality_final = (formality_total_text + formality_total_title) / 2
    
    return formality_final
    
def readability_fkgl_smog(title,text):
    """Determines the readability of the text based on the FKGL and SMOG calculations"""
    polysyllables = 0
    blob = textblob.TextBlob(title)
    total_words = len(blob.words)
    total_sentences = len(blob.sentences)
    total_syllables = syllables.estimate(str(blob))
    try:
        fkgl_value_title = ((.39 * (total_words/total_sentences)) + (11.8 * (total_syllables/total_words)) - 15.59)
    except:
        fkgl_value_title = 1
    
    split_sentence = title.split()
    
    for i in range(0,len(split_sentence)):
        if syllables.estimate(split_sentence[i]) >= 3:
            polysyllables += 1
    try:
        smog_calc_title = (1.043 * (math.sqrt(polysyllables * (30/total_sentences))) + 3.1291)
    except:
        smog_calc_title = 1
    
    blob1 = textblob.TextBlob(text)
    total_words = len(blob1.words)
    total_sentences = len(blob1.sentences)
    total_syllables = syllables.estimate(str(blob1))
    try:
        fkgl_value_text = ((.39 * (total_words/total_sentences)) + (11.8 * (total_syllables/total_words)) - 15.59)
    except:
        fkgl_value_text = 1
    
    split_sentence = title.split()
    
    for i in range(0,len(split_sentence)):
        if syllables.estimate(split_sentence[i]) >= 3:
            polysyllables += 1
    try:
        smog_calc_text = (1.043 * (math.sqrt(polysyllables * (30/total_sentences))) + 3.1291)
    except:
        smog_calc_text = 1
    
    fkgl_avg = (fkgl_value_title + fkgl_value_text) / 2
    
    smog_avg = (smog_calc_text + smog_calc_title) / 2
    
    return fkgl_avg, smog_avg
    
    
    
    
def main():
    """Main program that the user uses to determine Fake or True News."""
    fake_polarity = 0
    fake_subjectivity = 0
    fake_fkgl = 0
    fake_smog = 0
    fake_formality = 0
    true_polarity = 0
    true_subjectivity = 0
    true_formality = 0
    true_fkgl = 0
    true_smog = 0
    maintain = "yes"
    
    
    
    numbers_list, true_fake_number_list, fake_news_list, true_news_list = preprocess_data()
    
    try:
        clf = joblib.load("NN.joblib")
    
    except:
        
        clf = sklearn.neural_network.MLPClassifier()
        clf = clf.fit(numbers_list, true_fake_number_list)
        joblib.dump(clf, "NN.joblib")
    
    for row in fake_news_list:
        fake_polarity = fake_polarity + row[0]
        fake_subjectivity = fake_subjectivity + row[1]
        fake_formality = fake_formality + row[2]
        fake_fkgl = fake_fkgl + row[3]
        fake_smog = fake_smog + row[4]
    avg_polarity_fake = fake_polarity / len(fake_news_list)
    avg_subjectivity_fake = fake_subjectivity / len(fake_news_list)
    avg_formality_fake = fake_formality / len(fake_news_list)
    avg_fkgl_fake = fake_fkgl / len(fake_news_list)
    avg_smog_fake = fake_smog / len(fake_news_list)
    
    for row in true_news_list:
        true_polarity = true_polarity + row[0]
        true_subjectivity = true_subjectivity + row[1]
        true_formality = true_formality + row[2]
        true_fkgl = true_fkgl + row[3]
        true_smog = true_smog + row[4]
    avg_polarity_true = true_polarity / len(true_news_list)
    avg_subjectivity_true = true_subjectivity / len(true_news_list)
    avg_formality_true = true_formality / len(true_news_list)
    avg_fkgl_true = true_fkgl / len(true_news_list)
    avg_smog_true = true_smog / len(true_news_list)

    print("Welcome to the FND-3000 (Fake News Detector-3000)")
    
    
    
    while maintain.lower() == "yes":
        
        selection = input("Would you like to analyze one news article, a dataset of articles,\nor the accuracy of the FND-3000? (Single/Dataset/Accuracy): ")
        if selection.lower() == "single":
            
            pol = 0
            subject = 0
            formal = 0
            Fkgl = 0
            Smog = 0
            title = input("Please enter the title of the article: ")
            body_article = input("Please enter the body of the article: ")
            temp_list = []
            pol = polarity(title, body_article)
            subject = subjectivity(title, body_article)
            formal = formality(title, body_article)
            Fkgl , Smog = readability_fkgl_smog(title, body_article)
            
            temp_list.append([pol,subject,formal,Fkgl,Smog])
            
            determination = list(clf.predict(temp_list))
                    
                        
            false_or_true = determination.count(1)
            
            if false_or_true == 1:
                tkinter.messagebox.showinfo("Results","Based on previous data, the news article would be considered fake.")
            elif false_or_true == 0:
                tkinter.messagebox.showinfo("Results", "Based on previous data, the news article would be considered true.")
            
            
            #Reference: https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
            fig , graph = plt.subplots(5)
            
            graph[0].bar(["True","Your Article","Fake"],[avg_polarity_true,pol,avg_polarity_fake])
            graph[0].set_title("Average Polarity")
            graph[1].bar(["True","Your Article","Fake"],[avg_subjectivity_true,subject,avg_subjectivity_fake])
            graph[1].set_title("Average Subjectivity")
            graph[2].bar(["True","Your Article","Fake"],[avg_formality_true,formal,avg_formality_fake])
            graph[2].set_title("Average Formality")
            graph[3].bar(["True","Your Article","Fake"],[avg_fkgl_true,Fkgl,avg_fkgl_fake])
            graph[3].set_title("Average FKGL (Flesch-Kincaid Grade Level) Readability")
            graph[4].bar(["True","Your Article","Fake"],[avg_smog_true,Smog,avg_smog_fake])
            graph[4].set_title("Average SMOG Readability")
            plt.subplots_adjust(left = .13,right = .9,bottom = .05,top = .94,wspace = .91, hspace = .51)
            plt.show()
            
        elif selection.lower() == "dataset":    
            file_name = input("Please enter the name of the file with news you wish to analyze: ")
            try:
                with open(file_name,"r",encoding = "utf-8") as file:
                    reader = csv.reader(file)
                    new_values = []
                    pol_list = []
                    subject_list = []
                    formal_list = []
                    fkgl_list = []
                    smog_list = []
                    for row in reader:
                        pol = 0
                        subject = 0
                        formal = 0
                        Fkgl = 0
                        Smog = 0
                        
                        temp_list = []
                        
                        pol = polarity(row[0], row[1])
                        subject = subjectivity(row[0], row[1])
                        formal = formality(row[0], row[1])
                        Fkgl , Smog = readability_fkgl_smog(row[0], row[1])
                        
                        pol_list.append(pol)
                        subject_list.append(subject)
                        formal_list.append(formal)
                        fkgl_list.append(Fkgl)
                        smog_list.append(Smog)
                        
                        temp_list.extend([pol,subject,formal,Fkgl,Smog])
                        
                        new_values.append(temp_list)
                        
                    determination = list(clf.predict(new_values))
                    
                        
                false_or_true = (determination.count(1) / len(determination)) * 100
                
                if false_or_true  == 100:
                    tkinter.messagebox.showinfo("Results","Based on previous data, the news article(s) provided are " + str(false_or_true) + "% fake or completely fake.")
                elif false_or_true < 100 and false_or_true >= 80:
                    tkinter.messagebox.showinfo("Results","Based on previous data, the news article(s) provided are " + str(false_or_true) + "% fake or mostly fake.")
                elif false_or_true < 80 and false_or_true >= 60:
                    tkinter.messagebox.showinfo("Results", "Based on previous data, the news article(s) provided are " + str(false_or_true) + "% fake or partly fake.")
                elif false_or_true < 60 and false_or_true >= 40:
                    tkinter.messagebox.showinfo("Results", "Based on previous data, the news article(s) provided are " + str(false_or_true) + "% fake or neutral.")
                elif false_or_true < 40 and false_or_true >= 20:
                    tkinter.messagebox.showinfo("Results", "Based on previous data, the news article(s) provided are " + str(false_or_true) + "% fake or partly true.")
                elif false_or_true < 20 and false_or_true > 0:
                    tkinter.messagebox.showinfo("Results", "Based on previous data, the news article(s) provided are " + str(false_or_true) + "% fake or mostly true.")
                elif false_or_true == 0:
                    tkinter.messagebox.showinfo("Results", "Based on the previous data, the news articles provided are " + str(false_or_true) + "% fake or completely true.")
                
                
                #Reference: https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
                fig , graph = plt.subplots(5)
                
                graph[0].bar(["True","Your Dataset","Fake"],[avg_polarity_true,stats.mean(pol_list),avg_polarity_fake])
                graph[0].set_title("Average Polarity")
                graph[1].bar(["True","Your Dataset","Fake"],[avg_subjectivity_true,stats.mean(subject_list),avg_subjectivity_fake])
                graph[1].set_title("Average Subjectivity")
                graph[2].bar(["True","Your Dataset","Fake"],[avg_formality_true,stats.mean(formal_list),avg_formality_fake])
                graph[2].set_title("Average Formality")
                graph[3].bar(["True","Your Dataset","Fake"],[avg_fkgl_true,stats.mean(fkgl_list),avg_fkgl_fake])
                graph[3].set_title("Average FKGL (Flesch-Kincaid Grade Level) Readability")
                graph[4].bar(["True","Your Dataset","Fake"],[avg_smog_true,stats.mean(smog_list),avg_smog_fake])
                graph[4].set_title("Average SMOG Readability")
                plt.subplots_adjust(left = .13,right = .9,bottom = .05,top = .94,wspace = .91, hspace = .51)
                plt.show()
                    
                        
            except:
                tkinter.messagebox.showwarning("Error","Sorry, that file could not be found.")
                

        elif selection.lower() == "accuracy":
            number_fake = []
            predicted_false = None
            number_true = []
            predicted_true = None
            with open("Fake501-1500.csv","r",encoding = "utf-8") as file:
                reader = csv.reader(file)
                
                new_values = []
                pol_list = []
                subject_list = []
                formal_list = []
                fkgl_list = []
                smog_list = []
                for row in reader:
                    pol = 0
                    subject = 0
                    formal = 0
                    Fkgl = 0
                    Smog = 0
                    
                    temp_list = []
                    
                    pol = polarity(row[0], row[1])
                    subject = subjectivity(row[0], row[1])
                    formal = formality(row[0], row[1])
                    Fkgl , Smog = readability_fkgl_smog(row[0], row[1])
                    
                    pol_list.append(pol)
                    subject_list.append(subject)
                    formal_list.append(formal)
                    fkgl_list.append(Fkgl)
                    smog_list.append(Smog)
                    
                    temp_list.extend([pol,subject,formal,Fkgl,Smog])
                    
                    new_values.append(temp_list)
                    number_fake.append(0)
                    
                predicted_false = list(clf.predict(new_values))
                
            with open("True501-1500.csv","r",encoding = "utf-8") as file:
                
                reader = csv.reader(file)
                
                new_values = []
                pol_list = []
                subject_list = []
                formal_list = []
                fkgl_list = []
                smog_list = []
                for row in reader:
                    pol = 0
                    subject = 0
                    formal = 0
                    Fkgl = 0
                    Smog = 0
                    
                    temp_list = []
                    
                    pol = polarity(row[0], row[1])
                    subject = subjectivity(row[0], row[1])
                    formal = formality(row[0], row[1])
                    Fkgl , Smog = readability_fkgl_smog(row[0], row[1])
                    
                    pol_list.append(pol)
                    subject_list.append(subject)
                    formal_list.append(formal)
                    fkgl_list.append(Fkgl)
                    smog_list.append(Smog)
                    
                    temp_list.extend([pol,subject,formal,Fkgl,Smog])
                    
                    new_values.append(temp_list)
                    number_true.append(1) 
                
                predicted_true = list(clf.predict(new_values))
            
            predicted_total = predicted_false + predicted_true
            print(len(predicted_total))
            number_total = number_fake + number_true
            print(len(number_total))
            
            print(sklearn.metrics.classification_report(number_total,predicted_total,target_names = ["Fake News", "True News"]))

        
        else:
            print("Sorry, that is an invalid option. Please try again.")
            
        maintain = input("Would you like to make another selection? (yes/no): ")

if __name__ == "__main__":
    main()
from __future__ import print_function
import sys
import argparse
import collections
import pprint


def update_word_counter(matrix, counter, word_list, tag="CHEM"):
    for row in matrix:
        if "," in row:
            start, end = row.replace(tag, "")[1:-1].split(",")
        else:
            start = row.replace(tag, "")[1: -1]
            end = int(start) + 1
        counter.update({" ".join(word_list[int(start):int(end)+1]): 1})
    return counter

## input as sentence level labelst_ner
def get_ner_fmeasure(sentence_lists, golden_lists, predict_lists, label_type="BMES"):
    golden_counter = collections.Counter()
    predict_counter = collections.Counter()
    right_counter = collections.Counter()
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list, word_list)
            pred_matrix = get_ner_BIO(predict_list, word_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_counter = update_word_counter(gold_matrix, golden_counter, word_list)
        predict_counter = update_word_counter(pred_matrix, predict_counter, word_list)
        right_counter = update_word_counter(right_ner, right_counter, word_list)
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure, golden_counter, predict_counter, right_counter


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list, word_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix



def readSentence(input_file):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences,labels


def readTwoLabelSentence(input_file, golden_col, pred_col):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        pair = line.replace("\n", "").split("\t")
        if "##score##" in line:
            continue
        if len(pair) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            sentence.append(pair[0])
            golden_label.append(pair[golden_col])
            predict_label.append(pair[pred_col])

    return sentences,golden_labels,predict_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser("mertics")
    parser.add_argument("--schema", type=str, help="tag schema")
    parser.add_argument("--input-path", type=str, help="predicted path")
    parser.add_argument("--verbose-num", type=int, help="print golden and predict counter most common num")
    opt = parser.parse_args()

    sentences, golden_labels, predict_labels = readTwoLabelSentence(opt.input_path, golden_col=-2, pred_col=-1)
    acc, P, R, F, golden_counter, predict_counter, right_counter = get_ner_fmeasure(sentences, golden_labels, predict_labels, opt.schema.upper())
    print("============= golden ==================")
    pprint.pprint(golden_counter.most_common(opt.verbose_num))
    print("============= predict =================")
    pprint.pprint(predict_counter.most_common(opt.verbose_num))
    print("============= right =================")
    pprint.pprint(right_counter.most_common(opt.verbose_num))
    print ("Acc:%s P:%sm R:%s, F:%s"%(acc, P, R, F))

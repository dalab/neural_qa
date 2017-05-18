import numpy as np
import re
import unicodedata


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def file_len(fname):
    with open(fname, encoding="UTF-8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def load_q_t(filePath):
    with open(filePath, "r", encoding="UTF-8") as f:
        x_orig = []
        x_tree = []
        for line in f:
            row = [el.strip() for el in line.split("\t")]
            x_orig.append(strip_accents(row[1]))
            x_tree.append(strip_accents(row[2]))
    return [np.array(x_orig, dtype=np.object_), np.array(x_tree, dtype=np.object_)]


def load_training_data_efficient(filePath, vocab_processor):
    num_rows = file_len(filePath)
    x_q = np.zeros((num_rows, vocab_processor.max_document_length), dtype=np.int16)
    x_t = np.zeros((num_rows, vocab_processor.max_document_length), dtype=np.int16)
    y = np.zeros((num_rows, 2), dtype=np.int8)

    with open(filePath, "r", encoding="UTF-8") as f:
        ids = []
        for i, line in enumerate(f):
            row = [el.strip() for el in line.split("\t")]
            ids.append(strip_accents(row[0]))
            x_q[i] = np.array(list(vocab_processor.transform([strip_accents(row[1])])))
            x_t[i] = np.array(list(vocab_processor.transform([strip_accents(row[2])])))
            y[i] = (np.array([1, 0]) if row[3] == "false" else np.array([0, 1]))
    return [np.array(ids), x_q, x_t, y]


def load_training_data_efficient_with_add_features(filePath, vocab_processor):
    num_rows = file_len(filePath)
    x_q = np.zeros((num_rows, vocab_processor.max_document_length), dtype=np.int16)
    x_t = np.zeros((num_rows, vocab_processor.max_document_length), dtype=np.int16)
    features = np.zeros((num_rows, 5), dtype=np.int8)
    y = np.zeros((num_rows, 2), dtype=np.int8)

    with open(filePath, "r", encoding="UTF-8") as f:
        ids = []
        for i, line in enumerate(f):
            row = [el.strip() for el in line.split("\t")]
            ids.append(strip_accents(row[0]))
            tree = strip_accents(row[2])
            x_q[i] = np.array(list(vocab_processor.transform([strip_accents(row[1])])))
            x_t[i] = np.array(list(vocab_processor.transform([tree])))

            stripped_answer = strip_accents(row[4])
            answer_list = parse_answer(stripped_answer)
            a_type = answer_type(stripped_answer)

            multiple_feat = [1] if len(answer_list) > 1 else [0]
            ans_type_feat = [0, 0, 0]
            is_answer_cell = [1]

            if a_type == "number":
                ans_type_feat[0] = 1
                if not tree.startswith("numbernorm"):
                    is_answer_cell[0] = 0
            elif a_type == "date":
                ans_type_feat[1] = 1
            elif a_type == "name":
                ans_type_feat[2] = 1

            features[i] = np.array(multiple_feat + ans_type_feat + is_answer_cell)
            y[i] = (np.array([1, 0]) if row[3] == "false" else np.array([0, 1]))
    return [np.array(ids), x_q, x_t, y, features]


def load_training_data_efficient_with_answer(filePath, vocab_processor):
    num_rows = file_len(filePath)
    x_q = np.zeros((num_rows, vocab_processor.max_document_length), dtype=np.int16)
    x_t = np.zeros((num_rows, vocab_processor.max_document_length), dtype=np.int16)
    answers = np.zeros(num_rows, dtype=np.object_)
    y = np.zeros((num_rows, 2), dtype=np.int8)
    features = np.zeros((num_rows, 5), dtype=np.int8)

    with open(filePath, "r", encoding="UTF-8") as f:
        ids = []
        for i, line in enumerate(f):
            row = [el.strip() for el in line.split("\t")]
            ids.append(strip_accents(row[0]))
            tree = strip_accents(row[2])
            x_q[i] = np.array(list(vocab_processor.transform([strip_accents(row[1])])))
            x_t[i] = np.array(list(vocab_processor.transform([tree])))
            stripped_answer = strip_accents(row[4])
            answer_list = parse_answer(stripped_answer)
            a_type = answer_type(stripped_answer)

            multiple_feat = [1] if len(answer_list) > 1 else [0]
            ans_type_feat = [0, 0, 0]
            is_answer_cell = [1]

            if a_type == "number":
                ans_type_feat[0] = 1
                if not tree.startswith("numbernorm"):
                    is_answer_cell[0] = 0
            elif a_type == "date":
                ans_type_feat[1] = 1
            elif a_type == "name":
                ans_type_feat[2] = 1

            features[i] = np.array(multiple_feat + ans_type_feat + is_answer_cell)
            answers[i] = "\t".join(answer_list)
            y[i] = (np.array([1, 0]) if row[3] == "false" else np.array([0, 1]))
    return [np.array(ids), x_q, x_t, y, features, answers]


preg = re.compile(r'\(([^()]+)\)')


def my_parser(answer_string):
    level = 0
    matches = []
    record = False
    string_constructed = ""
    for c in answer_string:
        if c == ")":
            if level == 2:
                record = False
                matches.append(string_constructed)
                string_constructed = ""
            level -= 1
        if record:
            string_constructed += c
        if c == '(':
            level += 1
        if level == 2:
            record = True
    return matches


def answer_type(answer_string):
    matches = my_parser(answer_string)
    for m in matches:
        return m.split(" ")[0]


def parse_answer(answer_string):
    answer_list = []
    matches = my_parser(answer_string)
    for match in matches:
        answer = match.split(" ")
        if answer[0] == "number":
            answer_list.append(answer[1])
        elif answer[0] == "date":
            yy = "xx" if answer[1] == "-1" else answer[1][-4:]
            mm = "xx" if answer[2] == "-1" else ("0" + answer[2])[-2:]
            dd = "xx" if answer[3] == "-1" else ("0" + answer[3])[-2:]
            answer_list.append("{}-{}-{}".format(yy, mm, dd))
        elif answer[0] == "name":
            x = " ".join(answer[2:])
            if len(answer) > 3:
                x = x[1:len(x) - 1]
            if x.startswith('\\"') and x.endswith('\\"'):
                x = x[2:len(x) - 2]
            x = x.replace('\\n', " ")
            x = x.replace('\\"', '"')
            answer_list.append(x)
    return answer_list


def answer_to_official(answers, file_path):
    f = open(file_path, 'w', encoding="UTF-8")
    for id, a in answers:
        f.write(u"{}\t{}\n".format(id, a))
    f.close()


def read_question_id_dict(filePath):
    with open(filePath, "r", encoding="UTF-8") as f:
        i = 0
        id_question = {}
        for line in f:
            row = [el.strip() for el in line.split("\t")]
            if strip_accents(row[1]) in id_question:
                print(i)
            id_question[strip_accents(row[1])] = row[0]
            i += 1
    return id_question


def validation_questions(filePath):
    with open(filePath, "r", encoding="UTF-8") as f:
        questions = []
        for line in f:
            row = [el.strip() for el in line.split("\t")]
            if len(row) >= 1:
                questions.append(strip_accents(row[0]))

    return np.array(questions, dtype=np.object_)

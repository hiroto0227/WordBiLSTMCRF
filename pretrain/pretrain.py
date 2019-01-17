import numpy as np


def load_word2vec(word2vec_path):
    with open(word2vec_path, "r", errors="ignore") as f:
        word2vec = {}
        for line in f:
            if line:
                line_splited = line.split()
                word, vec = line_splited[0], line_splited[1]
                try:
                    word2vec[word] = np.array(vec, dtype=np.float32)
                except ValueError:
                    print(line)
                    # utf-8以外のwordが入ってくるので、エラーになってしまう。
                    print("ValueError: {}".format(word))
    return word2vec


def make_pretrain_embed(word2vec, token2id, word_embed_size):
    scale = np.sqrt(3/word_embed_size)
    embed = np.random.uniform(-scale, scale, [len(token2id), word_embed_size])
    perfect_match = 0
    case_match = 0
    not_match = 0
    error = 0
    for token, idx in token2id.items():
        try:
            embed[idx] = norm2vec(word2vec[token])
            perfect_match += 1
        except KeyError:
            try:
                embed[idx] = norm2vec(word2vec[token.lower()])
                case_match += 1
            except KeyError:
                print("OOV: {}".format(token))
                not_match += 1
            except ValueError:
                print("=" * 50)
                print(token.lower())
                print(word2vec[token.lower()].shape)
                error += 1
        except ValueError:
            print("=" * 50)
            print(token)
            print(word2vec[token].shape)
            error += 1
    print("perfect match: ",str(perfect_match)," case_match : ",str(case_match)," not match: ",str(not_match), "error: ",str(error))
    return embed


def norm2vec(vec, scale=3):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square / scale

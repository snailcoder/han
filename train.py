import tensorflow as tf
import numpy as np
import han_model
import data_utils
from datetime import datetime

tf.flags.DEFINE_string("review_path", "review.txt", "The yelp review file.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.flags.DEFINE_integer("gru_size", 50,
                        "The number of units int the GRU cell.")
tf.flags.DEFINE_integer("context_size", 80,
                        "The size of word/sentence context vectors.")
tf.flags.DEFINE_integer("min_count", 5,
                        "The word of which frequency is less than "
                        "min_count will be replaced with <UNK>.")
tf.flags.DEFINE_integer("max_doc_len", 15,
                        "The max sentence number contained in single "
                        "document.")
tf.flags.DEFINE_integer("max_sent_len", 100,
                        "The max word number contained in single sentence.")
tf.flags.DEFINE_integer("embedding_size", 200, "The word embedding size.")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size.")
tf.flags.DEFINE_integer("eval_step", 5,
                        "Evaluate model after this many steps.")
tf.flags.DEFINE_integer("early_stop_interval", 100,
                        "Stop optimizing if no improvement found in this many"
                        "epochs. Set this option 0 to disable early stopping.")
tf.flags.DEFINE_string("record_dir", "./tf_record", "Write logs to this dir.")
FLAGS = tf.flags.FLAGS

def main(_):
    data, vocab = data_utils.load_data(FLAGS.review_path)
    unk_vocab = data_utils.replace_UNK(vocab, FLAGS.min_count)
    word_idx_map = data_utils.get_word_idx_map(unk_vocab)
    all_text, all_label = zip(*data)
    labels = np.array(all_label)
    indexed_docs = data_utils.docs2mat(all_text, FLAGS.max_doc_len,
                                       FLAGS.max_sent_len, word_idx_map)
    folds = data_utils.split_train_test(indexed_docs, labels)
    # now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # checkpoint_dir = "{}/run-{}".format(FLAGS.record_dir, now)
    with tf.Graph().as_default():
         # Set log and checkpoint dir for saving and restoring model.
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True)
        sess = tf.Session(config=config)
        with sess.as_default():
            # Create essential model and operators.
            han =han_model.HanModel(FLAGS.max_doc_len, FLAGS.max_sent_len,
                                    len(word_idx_map), FLAGS.embedding_size,
                                    FLAGS.learning_rate, FLAGS.gru_size,
                                    FLAGS.context_size, FLAGS.gru_size,
                                    FLAGS.context_size, labels.shape[1])
            embedding = han.embedding_from_scratch()
            logits = han.inference(embedding)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            loss_op = han.loss(logits)
            train_op = han.training(loss_op, global_step)
            eval_op = han.evaluate(logits)
            # Merge all summaries.
            merged_summaries = tf.summary.merge_all()
            # Create summary writers for trainning and testing respectively.
            train_writer = tf.summary.FileWriter(
                FLAGS.record_dir + "/train", sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.record_dir + "/test")
            # Now feed the data.
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            for train_data, test_data in folds:
                test_X, test_y = zip(*test_data)
                num_batches = int(
                    np.ceil(len(train_data) / FLAGS.batch_size))
                best_eval_accuracy = 0.0
                last_improvement_epoch = 0
                for epoch in range(FLAGS.num_epochs):
                    shuffled_indices = np.random.permutation(len(train_data))
                    shuffled_data = train_data[shuffled_indices]
                    avg_loss = 0.0
                    for i in range(num_batches):
                        beg = i * FLAGS.batch_size
                        end = min((i + 1) * FLAGS.batch_size, len(train_data))
                        cur_batch = shuffled_data[beg : end]
                        X_batch, y_batch = zip(*cur_batch)
                        train_feed_dict = {han.input_X: np.array(X_batch),
                                           han.input_y: np.array(y_batch)}
                        summary, _, loss, step = sess.run(
                            [merged_summaries, train_op, loss_op, global_step],
                            feed_dict=train_feed_dict)
                        avg_loss += loss / num_batches
                        train_writer.add_summary(summary, step)
                    train_acc = 1 - avg_loss
                    if (epoch % FLAGS.eval_step == 0
                        or epoch == FLAGS.num_epochs - 1):
                        val_feed_dict = {han.input_X: np.array(test_X),
                                         han.input_y: np.array(test_y)}
                        summary, eval_acc, step = sess.run(
                            [merged_summaries, eval_op, global_step],
                            feed_dict=val_feed_dict)
                        print ("Epoch:%d, training accuracy:%f,"
                               " validation accuracy:%f"
                               % (epoch, train_acc, eval_acc))
                        test_writer.add_summary(summary, step)
                        if eval_acc > best_eval_accuracy:
                            best_eval_accuracy = eval_acc
                            last_improvement_epoch = epoch
                            saver.save(sess, FLAGS.record_dir,
                                       global_step=global_step)
                    if (FLAGS.early_stop_interval > 0
                        and last_improvement_epoch - epoch > FLAGS.early_stop_interval):
                        break
                break
                        

if __name__ == "__main__":
    tf.app.run()
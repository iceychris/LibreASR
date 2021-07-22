/*
LICENSE NOTICE:
This code builds upon the beam search implementation
of the SpeechBrain project (https://github.com/speechbrain/speechbrain).
Find its license in $(LIBREASR)/3rdparty/licenses/LICENSE.speechbrain

CHANGES:
Ported the python implementation
(https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/decoders/transducer.py)
from Python to C++
*/

// libtorch/torchscript
#include <torch/script.h>

// general
#include <string>
#include <iostream>
#include <memory>

// types
#include <tuple>
#include <optional>

// file streaming stuff
#include <fstream>
#include <sstream>
#include <cstdint>

// custom
#include <cstdio>
#include <stdexcept>
#include <array>


// syntactic sugar
#define MODULE torch::jit::script::Module
#define TENSOR at::Tensor
#define OTENSOR c10::optional<TENSOR>
// states
#define STATE std::vector<std::tuple<at::Tensor, at::Tensor>>
#define OSTATE c10::optional<STATE>
#define RAW_STATE c10::IValue
// network constants & types 
#define OUTPUT std::tuple<TENSOR, OSTATE>
#define ENCODER_DIM 1024
#define ENCODER_LAYERS 8
#define PREDICTOR_DIM 1024
#define PREDICTOR_LAYERS 2
#define MAX_ITERS 5
#define STACK 6
#define FEATURE_SZ 128
#define AUDIO_CHUNK_SZ 0.08
#define AUDIO_BUFFER 2

// encoder_state, predictor_input, predictor_state, predictor_output
#define TILE_STATE std::tuple<TENSOR, TENSOR>

// tokenizer settings
#define TOKENIZER_PATH "/root/.cache/LibreASR/libreasr/en-1.1.0/tokenizer.yttm-model"

// search settings
#define BEAM_WIDTH 8


void printShape(std::string msg, at::Tensor t) {
  // if (msg == "chunks[i]") {
  //   std::cout << "[shape] " << msg << " " << t.sizes() << "\n" << std::flush;
  //   std::cout << "  [mean] " << t.mean().item<float>() << "\n" << std::flush;
  //   std::cout << "  [std ] " << t.std().item<float>() << "\n" << std::flush;
  // }
}


int nearestMultiple(int x, int base) {
  if (x % base == 0) {
    return x;
  }
  return int(base * round(x / (base*1.0)) - base);
}


std::string exec(std::string cmd) {
    std::array<char, 512> buffer;
    std::string result;
    const char * command = cmd.c_str();
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command, "r"), pclose);
    if (!pipe) {
      throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      result += buffer.data();
    }
    return result;
}


std::string decode(std::vector<int> ids) {
  // convert to string
  std::ostringstream oss;
  if (!ids.empty())
  {
    // Convert all but the last element to avoid a trailing ","
    std::copy(ids.begin(), ids.end()-1,
        std::ostream_iterator<int>(oss, " "));

    // Now add the last element with no delimiter
    oss << ids.back();
  }
  auto rawIn = oss.str();
  std::string input1(rawIn.begin(), rawIn.end());
  const char * input = input1.c_str();

  // shell out
  // TODO: replace shelling out by using native C++ API...
  std::string pre = "echo \"";
  std::string post = "\" | yttm decode --model " TOKENIZER_PATH;
  std::string command = pre + input + post;
  std::string result = exec(command);
  return result;
}


STATE convertToState(RAW_STATE s) {
  STATE state;
  std::vector<c10::IValue> v = s.toList().vec();
  for(std::vector<c10::IValue>::iterator it = v.begin(); it != v.end(); ++it) {
    auto h = (*it).toTuple()->elements()[0].toTensor();
    auto c = (*it).toTuple()->elements()[1].toTensor();
	  state.push_back(std::make_tuple(h, c));
  }
  return state;
} 


torch::jit::script::Module loadModel(std::string path) {
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(path);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model " << path << "\n";
    // TODO: throw exception or sth here...
  }
  std::cout << "Model " << path << " loaded.\n";
  return module;
}


std::tuple<TENSOR, TENSOR, TENSOR, bool> fixTiling(TENSOR chunk, TENSOR lastX, TENSOR lastRemainder) {

  // save for later
  if (lastX.size(1) == 0) {
    lastX = chunk;
    return std::make_tuple(chunk, lastX, lastRemainder, true);
  }

  // merge
  std::vector<TENSOR> vec;
  vec.push_back(lastRemainder);
  vec.push_back(lastX.slice(1, 0, -1));
  vec.push_back(chunk.slice(1, 0, -1));
  TENSOR merged = torch::cat(vec, 1);

  // cut
  int T = merged.size(1);
  int nm = nearestMultiple(T, STACK);
  chunk = merged.slice(1, 0, nm);
  lastRemainder = merged.slice(1, nm, T);
  lastX = merged.slice(1, 0, 0);

  // stack & downsample
  chunk = chunk.unfold(1, STACK, STACK);
  return std::make_tuple(chunk, lastX, lastRemainder, false);
}


TENSOR inferPreprocessor(MODULE preprocessor, TENSOR chunk) {
  // pack up inputs
  std::vector<torch::jit::IValue> inputs;
  at::Tensor x = chunk.unsqueeze(0).unsqueeze(2).unsqueeze(3); 
  at::Tensor lengths = torch::ones({1});
  bool inference = true;
  lengths[0] = chunk.size(0);
  inputs.push_back(x);
  inputs.push_back(lengths);
  inputs.push_back(inference);
  
  // infer
  auto out = preprocessor.forward(inputs).toTuple()->elements();
  at::Tensor output = out[0].toTensor();
  printShape("prepro", output);
  return output;
}


OUTPUT inferEncoder(MODULE encoder, TENSOR x, OSTATE state) {
  // pack up inputs
  std::vector<torch::jit::IValue> inputs;
  at::Tensor lengths = torch::ones({1});
  lengths[0] = x.size(1);
  inputs.push_back(x);
  inputs.push_back(lengths);
  inputs.push_back(state);
  
  // infer
  auto out = encoder.forward(inputs).toTuple()->elements();
  at::Tensor output = out[0].toTensor();
  printShape("encoder", output);

  // convert
  auto new_state = c10::optional<STATE>(convertToState(out[1]));
  return std::make_tuple(output, new_state);
}


OUTPUT inferPredictor(MODULE predictor, TENSOR x, OSTATE state) {
  // pack up inputs
  std::vector<torch::jit::IValue> inputs;
  at::Tensor lengths = torch::ones({1});
  lengths[0] = x.size(1);
  inputs.push_back(x);
  inputs.push_back(lengths);
  inputs.push_back(state);
  
  // infer
  auto out = predictor.forward(inputs).toTuple()->elements();
  at::Tensor output = out[0].toTensor();
  printShape("predictor", output);

  // convert
  auto new_state = c10::optional<STATE>(convertToState(out[1]));
  return std::make_tuple(output, new_state);
}


TENSOR inferJoint(MODULE joint, TENSOR pred_state, TENSOR enc_state) {
  // pack up inputs
  std::vector<torch::jit::IValue> inputs;
  bool softmax = true;
  bool log = true;
  inputs.push_back(pred_state);
  inputs.push_back(enc_state);
  inputs.push_back(softmax);
  inputs.push_back(log);
  
  // infer
  auto output = joint.forward(inputs).toTensor();
  printShape("joint", output);
  return output;
}


class Hypothesis {
  public:
    std::vector<int> prediction;
    float logp_score;
    OSTATE hidden_dec = OSTATE();
    OSTATE hidden_lm = OSTATE();

    Hypothesis(std::vector<int> pred, float lp, OSTATE hd) : 
      prediction(pred),
      logp_score(lp),
      hidden_dec(hd) {}

    Hypothesis() {
      this->prediction = {2};
      this->logp_score = 0.0;
    }

    float score() {
      return this->logp_score / (this->prediction.size() * 1.0);
    }

    void print() {
      std::cout << " score: " << this->score() << ", pred: " << this->prediction << std::flush << "\n";
    }

    bool operator==(Hypothesis r) {
      bool a = this->prediction == r.prediction;
      bool b = this->logp_score == r.logp_score;
      bool c = true;
      if (this->hidden_dec.has_value() && r.hidden_dec.has_value()) {
        bool c = this->hidden_dec.has_value() == r.hidden_dec.has_value();
      }
      return a && b && c;
    }
};


Hypothesis best(std::vector<Hypothesis> hyps) {
  auto b = std::max_element(hyps.begin(), hyps.end(),
      [] (Hypothesis lhs, Hypothesis rhs) {
      return lhs.score() < rhs.score();
  });
  return *b;
}


class Greedysearch { 
  public:
    // config
    int blank_id = 0;
    int bos_id = 2;

    // state
    TENSOR blank = torch::zeros({1, 1});
    TENSOR input_PN;
    TENSOR output_PN;
    Hypothesis hyp;

    // modules
    MODULE predictor;
    MODULE joint;

    Greedysearch(MODULE p, MODULE j) : predictor(p),
      joint(j) {
      this->input_PN = (
          torch::ones({1, 1}, torch::kInt32)
      );
      this->input_PN[0][0] = this->bos_id;

      // run predictor for the first time
      auto predictor_tpl = inferPredictor(this->predictor, this->input_PN, this->hyp.hidden_dec);
      this->output_PN = std::get<0>(predictor_tpl);
      auto predictor_state = std::get<1>(predictor_tpl);
      this->hyp.hidden_dec = predictor_state;
    }

    ~Greedysearch() {
      // destructor
    }

    std::vector<int> step(TENSOR tn_output) {
      auto hyp = this->hyp;
      auto predictor_input = this->input_PN;
      auto predictor_output = this->output_PN;
      auto predictor_state = hyp.hidden_dec;

      // iterate over tokens
      for (int j = 0; j < MAX_ITERS; j++) {

        // run joint
        auto joint_out = inferJoint(this->joint, predictor_output, tn_output);

        // decode
        auto cls = joint_out.argmax(3)[0][0][0].item<int>();
        if (cls == 0) {
          // blank
          break;
        } else {
          // non-blank
          hyp.prediction.push_back(cls);
          predictor_input[0][0] = cls;

          // run predictor
          auto predictor_tpl = inferPredictor(this->predictor, predictor_input, predictor_state);
          predictor_output = std::get<0>(predictor_tpl);
          predictor_state = std::get<1>(predictor_tpl);
        }
      }

      this->output_PN = predictor_output;
      hyp.hidden_dec = predictor_state;
      this->hyp = hyp;
      return hyp.prediction;
    }
};


class Beamsearch { 
  public:
    // config
    int blank_id = 0;
    int bos_id = 2;
    int beam_size = BEAM_WIDTH;
    int nbest = 5;
    float lm_weight = 0.0;
    float state_beam = 2.3;
    float expand_beam = 2.3;

    // state
    TENSOR blank = torch::zeros({1, 1});
    TENSOR input_PN;
    std::vector<Hypothesis> hyp;
    std::vector<Hypothesis> beam_hyps;

    // modules
    MODULE predictor;
    MODULE joint;

    Beamsearch(MODULE p, MODULE j) : predictor(p),
      joint(j) {
      // start beam search
      // min between beam and max_target_lent
      // if we use RNN LM keep there hiddens
      // prepare BOS = Blank for the Prediction Network (PN)
      // Prepare Blank prediction
      this->input_PN = (
          torch::ones({1, 1}, torch::kInt32)
      );
      this->input_PN[0][0] = this->bos_id;

      // First forward-pass on PN
      auto pred = {this->bos_id};
      auto hyp = Hypothesis(pred, 0.0, OSTATE());
      this->hyp = {hyp};
      this->beam_hyps = {hyp};
    }

    ~Beamsearch() {
      // destructor
    }

    std::vector<int> step(TENSOR tn_output) {
      // get state
      blank = this->blank;
      input_PN = this->input_PN;
      hyp = this->hyp;
      beam_hyps = this->beam_hyps;
      std::vector<int> nbest_batch = {};
      std::vector<int> nbest_batch_score = {};

      // get hyps for extension
      auto process_hyps = beam_hyps;
      beam_hyps = {};

      while (true) {
        if (beam_hyps.size() >= this->beam_size) break;

        // Add norm score
        auto a_best_hyp = best(process_hyps);

        // Break if best_hyp in A is worse by more than state_beam than best_hyp in B
        if (beam_hyps.size() > 0) {
          auto b_best_hyp = best(beam_hyps); 
          auto a_best_prob = a_best_hyp.logp_score;
          auto b_best_prob = b_best_hyp.logp_score;
          if (b_best_prob >= (this->state_beam + a_best_prob)) break;
        }

        // remove best hyp from process_hyps
        process_hyps.erase(std::remove(process_hyps.begin(), process_hyps.end(), a_best_hyp), process_hyps.end());

        // forward PN
        input_PN[0][0] = a_best_hyp.prediction.back();
        auto predictor_tpl = inferPredictor(this->predictor, input_PN, a_best_hyp.hidden_dec);
        auto out_PN = std::get<0>(predictor_tpl);
        auto hidden = std::get<1>(predictor_tpl);

        // forward joint
        auto log_probs = inferJoint(this->joint, out_PN, tn_output);

        // if self.lm_weight > 0:
        //     log_probs_lm, hidden_lm = self._lm_forward_step(
        //         input_PN, a_best_hyp["hidden_lm"]
        //     )

        // Sort outputs at time
        auto res = log_probs.view(-1).topk(this->beam_size, -1, true, true);
        auto logp_targets = std::get<0>(res);
        auto positions = std::get<1>(res);
        float best_logp;
        if (positions[0].item<int>() != this->blank_id) {
          best_logp = logp_targets[0].item<float>();
        } else {
          best_logp = logp_targets[1].item<float>();
        }

        // Extend hyp by selection
        for (int j = 0; j < logp_targets.size(0); j++) {
          // hyp
          auto topk_hyp = Hypothesis(
            a_best_hyp.prediction,
            a_best_hyp.logp_score + logp_targets[j].item<float>(),
            a_best_hyp.hidden_dec
          );

          if (positions[j].item<int>() == this->blank_id) {
            beam_hyps.push_back(topk_hyp);
            // if self.lm_weight > 0:
            //     topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
            continue;
          }

          if (logp_targets[j].item<float>() >= (best_logp - this->expand_beam)) {
            topk_hyp.prediction.push_back(positions[j].item<int>());
            topk_hyp.hidden_dec = hidden;
            // if self.lm_weight > 0:
            //     topk_hyp["hidden_lm"] = hidden_lm
            //     topk_hyp["logp_score"] += (
            //         self.lm_weight
            //         * log_probs_lm[0, 0, positions[j]]
            //     )
            process_hyps.push_back(topk_hyp);
          }
        }
      }

      // save for use in self.forward_step(...)
      this->input_PN = input_PN;
      this->hyp = hyp;
      this->beam_hyps = beam_hyps;

      // grab best hypothesis
      auto vec = beam_hyps;
      std::sort(vec.begin(), vec.end(), 
        [] (Hypothesis lhs, Hypothesis rhs) {
        return lhs.score() > rhs.score();
      });
      auto currentBest = vec.front().prediction;

      // cut off bos
      currentBest = std::vector<int>(currentBest.begin() + 1, currentBest.end());
      return currentBest;
    }
};


class Chunker {
  public:
    // modules
    MODULE preprocessor;
    MODULE encoder;

    // state
    OSTATE encoder_state;
    TILE_STATE tile_state;

    Chunker(MODULE pre, MODULE e) : 
      preprocessor(pre),
      encoder(e) {
        this->encoder_state = OSTATE();
        auto lastX = torch::zeros({1, 0, FEATURE_SZ}, torch::kFloat32);
        auto lastRemainder = torch::zeros({1, 0, FEATURE_SZ}, torch::kFloat32);
        this->tile_state = std::make_tuple(lastX, lastRemainder);
      }

    std::vector<TENSOR> step(TENSOR chunk) {
      auto encoder_state = this->encoder_state;
      auto tile_state = this->tile_state;

      // [1, Ta] -> [1, Ts, Hp]
      auto preprocessed = inferPreprocessor(preprocessor, chunk);

      // slice up, stack & downsample
      auto lastX = std::get<0>(tile_state);
      auto lastRemainder = std::get<1>(tile_state);
      auto tiled_tpl = fixTiling(preprocessed, lastX, lastRemainder);
      auto encoder_input = std::get<0>(tiled_tpl);
      lastX = std::get<1>(tiled_tpl);
      lastRemainder = std::get<2>(tiled_tpl);
      auto skip = std::get<3>(tiled_tpl);

      // store state
      this->tile_state = std::make_tuple(lastX, lastRemainder);

      // do nothing
      if (skip) {
        return {};
      }

      // [1, Ts, Hp] -> [1, Ts, He]
      auto encoder_tpl = inferEncoder(encoder, encoder_input, encoder_state);
      auto encoder_out = std::get<0>(encoder_tpl);
      encoder_state = std::get<1>(encoder_tpl);
      this->encoder_state = encoder_state;

      // iterate over encoder outputs
      std::vector<TENSOR> vec = {};
      for (int i = 0; i < encoder_out.size(1); i++) {
        auto eout = encoder_out.slice(1, i, i+1);
        vec.push_back(eout);
      }
      return vec;
    }
};


int main(int argc, const char* argv[]) {

  // variables
  int sampleRate = 16000;
  float chunkSize = AUDIO_CHUNK_SZ * AUDIO_BUFFER;
  int typeSize = 4; // a float value takes 4 bytes
  int bufferSize = int(sampleRate * chunkSize * typeSize);
  int inputSize = int(sampleRate * chunkSize);

  // load all models
  auto preprocessor = loadModel("./models/preprocessor.pth");
  auto encoder = loadModel("./models/encoder.pth");
  auto predictor = loadModel("./models/predictor.pth");
  auto joint = loadModel("./models/joint.pth");

  // fire up stuff
  auto chunker = Chunker(preprocessor, encoder);
  auto searcher = Beamsearch(predictor, joint);
  // auto searcher = Greedysearch(predictor, joint);
  std::string sentence;

  // read input sequentially
  std::ifstream fin("/dev/stdin", std::ifstream::binary);
  std::vector<char> buffer(bufferSize, 0);
  while(!fin.eof()) {
    
    // read into buffer
    fin.read(buffer.data(), buffer.size());
    std::streamsize s=fin.gcount();
    // std::cout << "Read " << buffer.size() << " bytes.\n" << std::flush;

    // convert to tensor
    // TODO: use torch::from_blob(...) here
    auto chunk = torch::zeros(inputSize, torch::kFloat32);
    std::memcpy(chunk.data_ptr(), buffer.data(), sizeof(float) * chunk.numel());
    printShape("chunk", chunk);

    // do inference
    auto chunks = chunker.step(chunk);
    for (int i = 0; i < chunks.size(); i++) {
      printShape("chunks[i]", chunks[i]);
      auto tokens = searcher.step(chunks[i]);
      auto decoded = decode(tokens);
      if (sentence != decoded) {
        sentence = decoded;
        std::cout << sentence << std::flush;
      }
    }
  }
}

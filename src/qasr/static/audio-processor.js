class QASRPCM16Processor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options.processorOptions || {};
    this.targetRate = processorOptions.targetSampleRate || 16000;
    this.chunkFrames = processorOptions.chunkFrames || 1280;
    this.ratio = sampleRate / this.targetRate;
    this.sourceBuffer = [];
    this.sourcePosition = 0;
    this.outputBuffer = [];
    this.levelFrames = 0;
    this.levelEnergy = 0;
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === "flush") {
        this.flush();
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true;
    }

    const frames = input[0].length;
    for (let frame = 0; frame < frames; frame += 1) {
      let mono = 0;
      for (let channel = 0; channel < input.length; channel += 1) {
        mono += input[channel][frame] || 0;
      }
      mono /= input.length;
      this.sourceBuffer.push(mono);
      this.levelEnergy += mono * mono;
      this.levelFrames += 1;
    }

    this.resampleAvailable();
    if (this.levelFrames >= sampleRate / 10) {
      const rms = Math.sqrt(this.levelEnergy / this.levelFrames);
      this.port.postMessage({ type: "level", value: Math.min(1, rms * 5) });
      this.levelFrames = 0;
      this.levelEnergy = 0;
    }
    return true;
  }

  resampleAvailable() {
    while (this.sourcePosition + 1 < this.sourceBuffer.length) {
      const leftIndex = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - leftIndex;
      const left = this.sourceBuffer[leftIndex];
      const right = this.sourceBuffer[leftIndex + 1];
      this.outputBuffer.push(left + (right - left) * fraction);
      this.sourcePosition += this.ratio;

      if (this.outputBuffer.length >= this.chunkFrames) {
        this.emit(this.outputBuffer.splice(0, this.chunkFrames));
      }
    }

    const consumed = Math.floor(this.sourcePosition);
    if (consumed > 0) {
      this.sourceBuffer.splice(0, consumed);
      this.sourcePosition -= consumed;
    }
  }

  emit(samples) {
    if (samples.length === 0) {
      return;
    }
    const pcm = new Int16Array(samples.length);
    for (let index = 0; index < samples.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, samples[index]));
      pcm[index] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    this.port.postMessage({ type: "pcm", buffer: pcm.buffer }, [pcm.buffer]);
  }

  flush() {
    this.resampleAvailable();
    this.emit(this.outputBuffer.splice(0));
    this.port.postMessage({ type: "flushed" });
  }
}

registerProcessor("qasr-pcm16", QASRPCM16Processor);

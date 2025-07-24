package com.google.common.io;

import com.google.common.annotations.GwtIncompatible;
import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.BufferedOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;

@GwtIncompatible
/* loaded from: classes.dex */
public abstract class ByteSink {

    /* loaded from: classes.dex */
    public final class AsCharSink extends CharSink {
        private final Charset charset;

        @Override // com.google.common.io.CharSink
        public Writer openStream() {
            return new OutputStreamWriter(ByteSink.this.openStream(), this.charset);
        }

        public String toString() {
            return ByteSink.this.toString() + ".asCharSink(" + this.charset + ")";
        }

        private AsCharSink(Charset charset) {
            this.charset = (Charset) Preconditions.checkNotNull(charset);
        }
    }

    public CharSink asCharSink(Charset charset) {
        return new AsCharSink(charset);
    }

    public OutputStream openBufferedStream() {
        OutputStream openStream = openStream();
        return openStream instanceof BufferedOutputStream ? (BufferedOutputStream) openStream : new BufferedOutputStream(openStream);
    }

    public abstract OutputStream openStream();

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public void write(byte[] bArr) {
        Preconditions.checkNotNull(bArr);
        try {
            OutputStream outputStream = (OutputStream) Closer.create().register(openStream());
            outputStream.write(bArr);
            outputStream.flush();
        } finally {
        }
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    @CanIgnoreReturnValue
    public long writeFrom(InputStream inputStream) {
        Preconditions.checkNotNull(inputStream);
        try {
            OutputStream outputStream = (OutputStream) Closer.create().register(openStream());
            long copy = ByteStreams.copy(inputStream, outputStream);
            outputStream.flush();
            return copy;
        } finally {
        }
    }
}
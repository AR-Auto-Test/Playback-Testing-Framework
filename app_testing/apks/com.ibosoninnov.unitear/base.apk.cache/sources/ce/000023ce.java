package com.google.common.io;

import com.google.common.annotations.GwtIncompatible;
import com.google.common.base.Preconditions;
import java.io.IOException;
import java.io.Reader;
import java.nio.CharBuffer;

@GwtIncompatible
/* loaded from: classes.dex */
public final class CharSequenceReader extends Reader {
    private int mark;
    private int pos;
    private CharSequence seq;

    public CharSequenceReader(CharSequence charSequence) {
        this.seq = (CharSequence) Preconditions.checkNotNull(charSequence);
    }

    private void checkOpen() {
        if (this.seq == null) {
            throw new IOException("reader closed");
        }
    }

    private boolean hasRemaining() {
        return remaining() > 0;
    }

    private int remaining() {
        return this.seq.length() - this.pos;
    }

    @Override // java.io.Reader, java.io.Closeable, java.lang.AutoCloseable
    public synchronized void close() {
        this.seq = null;
    }

    @Override // java.io.Reader
    public synchronized void mark(int i) {
        Preconditions.checkArgument(i >= 0, "readAheadLimit (%s) may not be negative", i);
        checkOpen();
        this.mark = this.pos;
    }

    @Override // java.io.Reader
    public boolean markSupported() {
        return true;
    }

    @Override // java.io.Reader, java.lang.Readable
    public synchronized int read(CharBuffer charBuffer) {
        Preconditions.checkNotNull(charBuffer);
        checkOpen();
        if (hasRemaining()) {
            int min = Math.min(charBuffer.remaining(), remaining());
            for (int i = 0; i < min; i++) {
                CharSequence charSequence = this.seq;
                int i2 = this.pos;
                this.pos = i2 + 1;
                charBuffer.put(charSequence.charAt(i2));
            }
            return min;
        }
        return -1;
    }

    @Override // java.io.Reader
    public synchronized boolean ready() {
        checkOpen();
        return true;
    }

    @Override // java.io.Reader
    public synchronized void reset() {
        checkOpen();
        this.pos = this.mark;
    }

    @Override // java.io.Reader
    public synchronized long skip(long j) {
        int min;
        Preconditions.checkArgument(j >= 0, "n (%s) may not be negative", j);
        checkOpen();
        min = (int) Math.min(remaining(), j);
        this.pos += min;
        return min;
    }

    @Override // java.io.Reader
    public synchronized int read() {
        char c2;
        checkOpen();
        if (hasRemaining()) {
            CharSequence charSequence = this.seq;
            int i = this.pos;
            this.pos = i + 1;
            c2 = charSequence.charAt(i);
        } else {
            c2 = 65535;
        }
        return c2;
    }

    @Override // java.io.Reader
    public synchronized int read(char[] cArr, int i, int i2) {
        Preconditions.checkPositionIndexes(i, i + i2, cArr.length);
        checkOpen();
        if (hasRemaining()) {
            int min = Math.min(i2, remaining());
            for (int i3 = 0; i3 < min; i3++) {
                CharSequence charSequence = this.seq;
                int i4 = this.pos;
                this.pos = i4 + 1;
                cArr[i + i3] = charSequence.charAt(i4);
            }
            return min;
        }
        return -1;
    }
}
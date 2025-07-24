package com.google.common.io;

import com.google.common.annotations.GwtIncompatible;
import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.BufferedWriter;
import java.io.Writer;

@GwtIncompatible
/* loaded from: classes.dex */
public abstract class CharSink {
    public Writer openBufferedStream() {
        Writer openStream = openStream();
        return openStream instanceof BufferedWriter ? (BufferedWriter) openStream : new BufferedWriter(openStream);
    }

    public abstract Writer openStream();

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public void write(CharSequence charSequence) {
        Preconditions.checkNotNull(charSequence);
        try {
            Writer writer = (Writer) Closer.create().register(openStream());
            writer.append(charSequence);
            writer.flush();
        } finally {
        }
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    @CanIgnoreReturnValue
    public long writeFrom(Readable readable) {
        Preconditions.checkNotNull(readable);
        try {
            Writer writer = (Writer) Closer.create().register(openStream());
            long copy = CharStreams.copy(readable, writer);
            writer.flush();
            return copy;
        } finally {
        }
    }

    public void writeLines(Iterable<? extends CharSequence> iterable) {
        writeLines(iterable, System.getProperty("line.separator"));
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public void writeLines(Iterable<? extends CharSequence> iterable, String str) {
        Preconditions.checkNotNull(iterable);
        Preconditions.checkNotNull(str);
        try {
            Writer writer = (Writer) Closer.create().register(openBufferedStream());
            for (CharSequence charSequence : iterable) {
                writer.append(charSequence).append((CharSequence) str);
            }
            writer.flush();
        } finally {
        }
    }
}
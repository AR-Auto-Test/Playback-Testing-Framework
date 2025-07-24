package com.google.firebase.encoders;

/* loaded from: classes2.dex */
public interface Encoder<TValue, TContext> {
    void encode(TValue tvalue, TContext tcontext);
}
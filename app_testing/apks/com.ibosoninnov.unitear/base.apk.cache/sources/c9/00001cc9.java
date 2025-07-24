package com.google.ar.sceneform.rendering;

import android.util.Log;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Function;

/* loaded from: classes.dex */
public class FutureHelper {
    private FutureHelper() {
    }

    public static <T> CompletableFuture<T> logOnException(final String str, CompletableFuture<T> completableFuture, final String str2) {
        completableFuture.exceptionally((Function) new Function() { // from class: c.d.b.a.q.c
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Log.e(str, str2, th);
                throw new CompletionException(th);
            }
        });
        return completableFuture;
    }
}
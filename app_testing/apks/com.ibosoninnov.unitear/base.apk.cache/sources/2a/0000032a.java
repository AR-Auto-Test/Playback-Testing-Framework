package b.d.b.d1.k1.c;

import com.google.common.util.concurrent.ListenableFuture;

/* compiled from: AsyncFunction.java */
@FunctionalInterface
/* loaded from: classes.dex */
public interface b<I, O> {
    ListenableFuture<O> apply(I i);
}
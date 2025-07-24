package b.d.b.d1.k1.c;

import com.google.common.util.concurrent.ListenableFuture;

/* compiled from: Futures.java */
/* loaded from: classes.dex */
public class f implements b<I, O> {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ b.c.a.c.a f1546a;

    public f(b.c.a.c.a aVar) {
        this.f1546a = aVar;
    }

    @Override // b.d.b.d1.k1.c.b
    public ListenableFuture<O> apply(I i) {
        return g.c(this.f1546a.apply(i));
    }
}
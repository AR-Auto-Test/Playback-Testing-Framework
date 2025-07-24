package b.d.a.e;

import b.d.a.e.h1;
import java.util.LinkedHashSet;
import java.util.Objects;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class y implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ h1.a f1239b;

    @Override // java.lang.Runnable
    public final void run() {
        h1.a aVar = this.f1239b;
        Objects.requireNonNull(aVar);
        LinkedHashSet<p1> linkedHashSet = new LinkedHashSet();
        synchronized (aVar.f1067a.f1061b) {
            linkedHashSet.addAll(new LinkedHashSet(aVar.f1067a.f1064e));
            linkedHashSet.addAll(new LinkedHashSet(aVar.f1067a.f1062c));
        }
        for (p1 p1Var : linkedHashSet) {
            p1Var.b().m(p1Var);
        }
    }
}
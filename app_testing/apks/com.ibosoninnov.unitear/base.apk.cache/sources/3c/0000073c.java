package c.c.a.m.v;

import c.c.a.m.v.a;
import java.util.Objects;

/* compiled from: ActiveResources.java */
/* loaded from: classes.dex */
public class b implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ a f3598b;

    public b(a aVar) {
        this.f3598b = aVar;
    }

    @Override // java.lang.Runnable
    public void run() {
        a aVar = this.f3598b;
        Objects.requireNonNull(aVar);
        while (true) {
            try {
                aVar.b((a.b) aVar.f3590d.remove());
            } catch (InterruptedException unused) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
package b.j.g;

import b.j.d.d;

/* compiled from: CallbackWithHandler.java */
/* loaded from: classes.dex */
public class b implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ m f2124b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int f2125c;

    public b(c cVar, m mVar, int i) {
        this.f2124b = mVar;
        this.f2125c = i;
    }

    @Override // java.lang.Runnable
    public void run() {
        m mVar = this.f2124b;
        int i = this.f2125c;
        b.j.c.b.e eVar = ((d.a) mVar).f2104a;
        if (eVar != null) {
            eVar.onFontRetrievalFailed(i);
        }
    }
}
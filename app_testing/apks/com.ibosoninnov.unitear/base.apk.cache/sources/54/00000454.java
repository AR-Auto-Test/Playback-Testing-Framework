package b.j.g;

import android.graphics.Typeface;
import b.j.d.d;

/* compiled from: CallbackWithHandler.java */
/* loaded from: classes.dex */
public class a implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ m f2122b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Typeface f2123c;

    public a(c cVar, m mVar, Typeface typeface) {
        this.f2122b = mVar;
        this.f2123c = typeface;
    }

    @Override // java.lang.Runnable
    public void run() {
        m mVar = this.f2122b;
        Typeface typeface = this.f2123c;
        b.j.c.b.e eVar = ((d.a) mVar).f2104a;
        if (eVar != null) {
            eVar.onFontRetrieved(typeface);
        }
    }
}
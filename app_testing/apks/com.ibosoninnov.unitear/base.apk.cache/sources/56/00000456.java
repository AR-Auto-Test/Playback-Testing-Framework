package b.j.g;

import android.os.Handler;
import b.j.g.j;

/* compiled from: CallbackWithHandler.java */
/* loaded from: classes.dex */
public class c {

    /* renamed from: a  reason: collision with root package name */
    public final m f2126a;

    /* renamed from: b  reason: collision with root package name */
    public final Handler f2127b;

    public c(m mVar, Handler handler) {
        this.f2126a = mVar;
        this.f2127b = handler;
    }

    public void a(j.a aVar) {
        int i = aVar.f2149b;
        if (i == 0) {
            this.f2127b.post(new a(this, this.f2126a, aVar.f2148a));
            return;
        }
        this.f2127b.post(new b(this, this.f2126a, i));
    }
}
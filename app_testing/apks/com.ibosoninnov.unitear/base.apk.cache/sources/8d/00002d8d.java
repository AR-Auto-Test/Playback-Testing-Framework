package h.a.a;

import pl.droidsonroids.gif.GifInfoHandle;

/* compiled from: GifDrawable.java */
/* loaded from: classes2.dex */
public class b extends k {

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ c f6225c;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public b(c cVar, c cVar2) {
        super(cVar2);
        this.f6225c = cVar;
    }

    @Override // h.a.a.k
    public void a() {
        boolean reset;
        GifInfoHandle gifInfoHandle = this.f6225c.f6232h;
        synchronized (gifInfoHandle) {
            reset = GifInfoHandle.reset(gifInfoHandle.f6268b);
        }
        if (reset) {
            this.f6225c.start();
        }
    }
}
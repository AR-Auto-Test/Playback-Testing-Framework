package f.g0.i;

import f.g0.i.s;
import java.io.IOException;
import java.util.Objects;

/* compiled from: Http2Connection.java */
/* loaded from: classes2.dex */
public class j extends f.g0.b {

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int f5947c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ g.e f5948d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ int f5949e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ boolean f5950f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ g f5951g;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public j(g gVar, String str, Object[] objArr, int i, g.e eVar, int i2, boolean z) {
        super(str, objArr);
        this.f5951g = gVar;
        this.f5947c = i;
        this.f5948d = eVar;
        this.f5949e = i2;
        this.f5950f = z;
    }

    @Override // f.g0.b
    public void a() {
        try {
            s sVar = this.f5951g.l;
            g.e eVar = this.f5948d;
            int i = this.f5949e;
            Objects.requireNonNull((s.a) sVar);
            eVar.c(i);
            this.f5951g.t.H(this.f5947c, b.CANCEL);
            synchronized (this.f5951g) {
                this.f5951g.v.remove(Integer.valueOf(this.f5947c));
            }
        } catch (IOException unused) {
        }
    }
}
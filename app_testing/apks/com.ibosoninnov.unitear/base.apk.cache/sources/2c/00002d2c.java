package f.g0.i;

import f.g0.i.g;
import java.io.IOException;

/* compiled from: Http2Connection.java */
/* loaded from: classes2.dex */
public class n extends f.g0.b {

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ t f5958c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ g.f f5959d;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public n(g.f fVar, String str, Object[] objArr, t tVar) {
        super(str, objArr);
        this.f5959d = fVar;
        this.f5958c = tVar;
    }

    @Override // f.g0.b
    public void a() {
        try {
            g.this.t.B(this.f5958c);
        } catch (IOException unused) {
            g.B(g.this);
        }
    }
}
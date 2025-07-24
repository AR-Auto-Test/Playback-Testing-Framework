package f.g0.i;

import f.g0.i.g;
import java.io.IOException;

/* compiled from: Http2Connection.java */
/* loaded from: classes2.dex */
public class l extends f.g0.b {

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ p f5955c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ g.f f5956d;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public l(g.f fVar, String str, Object[] objArr, p pVar) {
        super(str, objArr);
        this.f5956d = fVar;
        this.f5955c = pVar;
    }

    @Override // f.g0.b
    public void a() {
        try {
            g.this.f5916d.b(this.f5955c);
        } catch (IOException e2) {
            f.g0.j.f fVar = f.g0.j.f.f6032a;
            StringBuilder x = c.b.a.a.a.x("Http2Connection.Listener failure for ");
            x.append(g.this.f5918f);
            fVar.k(4, x.toString(), e2);
            try {
                this.f5955c.c(b.PROTOCOL_ERROR);
            } catch (IOException unused) {
            }
        }
    }
}
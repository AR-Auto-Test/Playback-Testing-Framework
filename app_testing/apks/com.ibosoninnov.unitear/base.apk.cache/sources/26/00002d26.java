package f.g0.i;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/* compiled from: Http2Connection.java */
/* loaded from: classes2.dex */
public class h extends f.g0.b {

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int f5940c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ List f5941d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ g f5942e;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public h(g gVar, String str, Object[] objArr, int i, List list) {
        super(str, objArr);
        this.f5942e = gVar;
        this.f5940c = i;
        this.f5941d = list;
    }

    @Override // f.g0.b
    public void a() {
        Objects.requireNonNull(this.f5942e.l);
        try {
            this.f5942e.t.H(this.f5940c, b.CANCEL);
            synchronized (this.f5942e) {
                this.f5942e.v.remove(Integer.valueOf(this.f5940c));
            }
        } catch (IOException unused) {
        }
    }
}
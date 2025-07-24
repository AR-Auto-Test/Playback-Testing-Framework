package f.g0.i;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/* compiled from: Http2Connection.java */
/* loaded from: classes2.dex */
public class i extends f.g0.b {

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int f5943c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ List f5944d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ boolean f5945e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ g f5946f;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public i(g gVar, String str, Object[] objArr, int i, List list, boolean z) {
        super(str, objArr);
        this.f5946f = gVar;
        this.f5943c = i;
        this.f5944d = list;
        this.f5945e = z;
    }

    @Override // f.g0.b
    public void a() {
        Objects.requireNonNull(this.f5946f.l);
        try {
            this.f5946f.t.H(this.f5943c, b.CANCEL);
            synchronized (this.f5946f) {
                this.f5946f.v.remove(Integer.valueOf(this.f5943c));
            }
        } catch (IOException unused) {
        }
    }
}
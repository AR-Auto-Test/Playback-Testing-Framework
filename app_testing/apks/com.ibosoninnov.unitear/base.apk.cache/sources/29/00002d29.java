package f.g0.i;

import f.g0.i.s;
import java.util.Objects;

/* compiled from: Http2Connection.java */
/* loaded from: classes2.dex */
public class k extends f.g0.b {

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int f5952c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ b f5953d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ g f5954e;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public k(g gVar, String str, Object[] objArr, int i, b bVar) {
        super(str, objArr);
        this.f5954e = gVar;
        this.f5952c = i;
        this.f5953d = bVar;
    }

    @Override // f.g0.b
    public void a() {
        Objects.requireNonNull((s.a) this.f5954e.l);
        synchronized (this.f5954e) {
            this.f5954e.v.remove(Integer.valueOf(this.f5952c));
        }
    }
}
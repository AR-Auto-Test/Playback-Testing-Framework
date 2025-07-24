package c.e.b;

import android.util.Log;
import java.io.IOException;
import java.util.Objects;

/* compiled from: HttpHelperPost.java */
/* loaded from: classes2.dex */
public class dc implements f.e {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ec f4648a;

    public dc(ec ecVar) {
        this.f4648a = ecVar;
    }

    @Override // f.e
    public void a(f.d dVar, f.b0 b0Var) {
        if (!b0Var.B()) {
            this.f4648a.f4692c = "";
            StringBuilder x = c.b.a.a.a.x("Failed ");
            x.append(b0Var.toString());
            Log.d("HttpHelperPost", x.toString());
            return;
        }
        this.f4648a.f4692c = b0Var.f5730h.F();
        ec ecVar = this.f4648a;
        ecVar.f4693d.b(ecVar.f4692c);
    }

    @Override // f.e
    public void b(f.d dVar, IOException iOException) {
        this.f4648a.f4693d.a(iOException.getMessage());
        Objects.requireNonNull(this.f4648a);
        Log.d("HttpHelperPost", iOException.toString());
    }
}
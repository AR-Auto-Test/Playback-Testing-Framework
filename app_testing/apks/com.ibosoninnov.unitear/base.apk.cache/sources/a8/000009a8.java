package c.e.b;

import android.util.Log;
import java.io.IOException;
import java.util.Objects;

/* compiled from: HttpHelper.java */
/* loaded from: classes2.dex */
public class bc implements f.e {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ cc f4577a;

    public bc(cc ccVar) {
        this.f4577a = ccVar;
    }

    @Override // f.e
    public void a(f.d dVar, f.b0 b0Var) {
        if (!b0Var.B()) {
            cc ccVar = this.f4577a;
            ccVar.f4615c = "";
            ccVar.f4616d.a(b0Var.f5727e);
            Log.e("HttpHelper", "Failed " + b0Var);
            return;
        }
        cc ccVar2 = this.f4577a;
        f.d0 d0Var = b0Var.f5730h;
        Objects.requireNonNull(d0Var);
        ccVar2.f4615c = d0Var.F();
        cc ccVar3 = this.f4577a;
        ccVar3.f4616d.b(ccVar3.f4615c);
    }

    @Override // f.e
    public void b(f.d dVar, IOException iOException) {
        this.f4577a.f4616d.a(iOException.getMessage());
        Log.d("HttpHelper", iOException.toString());
    }
}
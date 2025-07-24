package c.e.b;

import android.util.Log;
import c.e.b.vc;
import com.ibosoninnov.unitear.ARCoreSceneformActivity;
import java.util.Objects;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class g implements vc.c {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ARCoreSceneformActivity f4760a;

    /* JADX DEBUG: Marked for inline */
    /* JADX DEBUG: Method not inlined, still used in: [c.e.b.r.run():void] */
    public /* synthetic */ g(ARCoreSceneformActivity aRCoreSceneformActivity) {
        this.f4760a = aRCoreSceneformActivity;
    }

    public final void a(String str) {
        Objects.requireNonNull(this.f4760a);
        Log.d("ARCoreSceneformActivity", str);
    }
}
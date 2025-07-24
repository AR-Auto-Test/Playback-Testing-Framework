package b.w.b;

import android.content.Context;
import android.util.DisplayMetrics;
import android.view.View;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: PagerSnapHelper.java */
/* loaded from: classes.dex */
public class t extends o {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ u f2797a;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public t(u uVar, Context context) {
        super(context);
        this.f2797a = uVar;
    }

    @Override // b.w.b.o
    public float calculateSpeedPerPixel(DisplayMetrics displayMetrics) {
        return 100.0f / displayMetrics.densityDpi;
    }

    @Override // b.w.b.o
    public int calculateTimeForScrolling(int i) {
        return Math.min(100, super.calculateTimeForScrolling(i));
    }

    @Override // b.w.b.o, androidx.recyclerview.widget.RecyclerView.z
    public void onTargetFound(View view, RecyclerView.a0 a0Var, RecyclerView.z.a aVar) {
        u uVar = this.f2797a;
        int[] a2 = uVar.a(uVar.f2803a.getLayoutManager(), view);
        int i = a2[0];
        int i2 = a2[1];
        int calculateTimeForDeceleration = calculateTimeForDeceleration(Math.max(Math.abs(i), Math.abs(i2)));
        if (calculateTimeForDeceleration > 0) {
            aVar.b(i, i2, calculateTimeForDeceleration, this.mDecelerateInterpolator);
        }
    }
}
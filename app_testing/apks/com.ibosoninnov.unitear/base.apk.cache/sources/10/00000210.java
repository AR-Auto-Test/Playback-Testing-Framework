package b.c0.a;

import android.graphics.Rect;
import android.view.View;
import androidx.viewpager.widget.ViewPager;
import b.j.j.j;
import b.j.j.q;
import b.j.j.w;

/* compiled from: ViewPager.java */
/* loaded from: classes.dex */
public class b implements j {

    /* renamed from: a  reason: collision with root package name */
    public final Rect f1006a = new Rect();

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ViewPager f1007b;

    public b(ViewPager viewPager) {
        this.f1007b = viewPager;
    }

    @Override // b.j.j.j
    public w onApplyWindowInsets(View view, w wVar) {
        w j = q.j(view, wVar);
        if (j.g()) {
            return j;
        }
        Rect rect = this.f1006a;
        rect.left = j.c();
        rect.top = j.e();
        rect.right = j.d();
        rect.bottom = j.b();
        int childCount = this.f1007b.getChildCount();
        for (int i = 0; i < childCount; i++) {
            w c2 = q.c(this.f1007b.getChildAt(i), j);
            rect.left = Math.min(c2.c(), rect.left);
            rect.top = Math.min(c2.e(), rect.top);
            rect.right = Math.min(c2.d(), rect.right);
            rect.bottom = Math.min(c2.b(), rect.bottom);
        }
        return j.h(rect.left, rect.top, rect.right, rect.bottom);
    }
}
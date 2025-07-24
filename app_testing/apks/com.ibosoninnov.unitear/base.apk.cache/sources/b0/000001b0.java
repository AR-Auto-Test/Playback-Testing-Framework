package b.b.h;

import android.content.Context;
import android.graphics.Rect;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.TextView;
import com.ibosoninnov.unitear.R;

/* compiled from: TooltipPopup.java */
/* loaded from: classes.dex */
public class c1 {

    /* renamed from: a  reason: collision with root package name */
    public final Context f815a;

    /* renamed from: b  reason: collision with root package name */
    public final View f816b;

    /* renamed from: c  reason: collision with root package name */
    public final TextView f817c;

    /* renamed from: d  reason: collision with root package name */
    public final WindowManager.LayoutParams f818d;

    /* renamed from: e  reason: collision with root package name */
    public final Rect f819e;

    /* renamed from: f  reason: collision with root package name */
    public final int[] f820f;

    /* renamed from: g  reason: collision with root package name */
    public final int[] f821g;

    public c1(Context context) {
        WindowManager.LayoutParams layoutParams = new WindowManager.LayoutParams();
        this.f818d = layoutParams;
        this.f819e = new Rect();
        this.f820f = new int[2];
        this.f821g = new int[2];
        this.f815a = context;
        View inflate = LayoutInflater.from(context).inflate(R.layout.abc_tooltip, (ViewGroup) null);
        this.f816b = inflate;
        this.f817c = (TextView) inflate.findViewById(R.id.message);
        layoutParams.setTitle(c1.class.getSimpleName());
        layoutParams.packageName = context.getPackageName();
        layoutParams.type = 1002;
        layoutParams.width = -2;
        layoutParams.height = -2;
        layoutParams.format = -3;
        layoutParams.windowAnimations = 2131886085;
        layoutParams.flags = 24;
    }

    public void a() {
        if (this.f816b.getParent() != null) {
            ((WindowManager) this.f815a.getSystemService("window")).removeView(this.f816b);
        }
    }
}
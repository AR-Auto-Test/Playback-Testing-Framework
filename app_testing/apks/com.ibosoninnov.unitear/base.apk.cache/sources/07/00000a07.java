package c.e.b.ff;

import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.databinding.ViewDataBinding;

/* compiled from: ItemArGalleryBinding.java */
/* loaded from: classes2.dex */
public abstract class c extends ViewDataBinding {
    public final TextView A;
    public final RelativeLayout B;
    public final ImageView r;
    public final ImageView s;
    public final ImageView t;
    public final ImageView u;
    public final ImageView v;
    public final LinearLayout w;
    public final LinearLayout x;
    public final LinearLayout y;
    public final TextView z;

    public c(Object obj, View view, int i, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, LinearLayout linearLayout, LinearLayout linearLayout2, LinearLayout linearLayout3, TextView textView, TextView textView2, RelativeLayout relativeLayout) {
        super(obj, view, i);
        this.r = imageView;
        this.s = imageView2;
        this.t = imageView3;
        this.u = imageView4;
        this.v = imageView5;
        this.w = linearLayout;
        this.x = linearLayout2;
        this.y = linearLayout3;
        this.z = textView;
        this.A = textView2;
        this.B = relativeLayout;
    }

    public abstract void m(c.e.b.hf.b bVar);
}
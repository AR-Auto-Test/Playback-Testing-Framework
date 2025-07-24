package c.e.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import com.ibosoninnov.unitear.LoginWebviewActivity;

/* compiled from: LoginWebviewActivity.java */
/* loaded from: classes2.dex */
public class be extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ LoginWebviewActivity f4580a;

    public be(LoginWebviewActivity loginWebviewActivity) {
        this.f4580a = loginWebviewActivity;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        super.onAnimationEnd(animator);
        this.f4580a.u.setVisibility(0);
    }
}
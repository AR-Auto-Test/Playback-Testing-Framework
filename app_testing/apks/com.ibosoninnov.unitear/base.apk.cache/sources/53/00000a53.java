package c.e.b.p000if;

import com.google.android.filament.gltfio.Animator;

/* compiled from: AnimationInstance.java */
/* renamed from: c.e.b.if.c  reason: invalid package */
/* loaded from: classes2.dex */
public class c {

    /* renamed from: a  reason: collision with root package name */
    public Animator f4867a;

    /* renamed from: b  reason: collision with root package name */
    public Long f4868b;

    /* renamed from: c  reason: collision with root package name */
    public float f4869c;

    /* renamed from: d  reason: collision with root package name */
    public int f4870d;

    public c(Animator animator, int i, Long l) {
        this.f4867a = animator;
        this.f4868b = l;
        this.f4869c = animator.getAnimationDuration(i);
        this.f4870d = i;
    }
}
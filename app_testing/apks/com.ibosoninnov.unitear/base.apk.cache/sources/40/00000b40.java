package c.e.b;

import android.media.MediaPlayer;
import java.util.function.BiConsumer;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class w5 implements BiConsumer {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ w5 f5371a = new w5();

    @Override // java.util.function.BiConsumer
    public final void accept(Object obj, Object obj2) {
        MediaPlayer mediaPlayer = (MediaPlayer) obj;
        if (((Boolean) obj2).booleanValue()) {
            mediaPlayer.start();
        }
    }
}
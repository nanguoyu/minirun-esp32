#import <Cocoa/Cocoa.h>
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2)
            return 2;
        NSString *text = [NSString stringWithUTF8String:argv[1]];
        if (!text)
            return 2;
        NSBitmapImageRep *rep =
            [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:NULL
                                                    pixelsWide:240
                                                    pixelsHigh:144
                                                 bitsPerSample:8
                                               samplesPerPixel:4
                                                      hasAlpha:YES
                                                      isPlanar:NO
                                                colorSpaceName:NSDeviceRGBColorSpace
                                                   bytesPerRow:240 * 4
                                                  bitsPerPixel:32];
        NSGraphicsContext *ctx = [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
        [NSGraphicsContext saveGraphicsState];
        [NSGraphicsContext setCurrentContext:ctx];
        [[NSColor blackColor] setFill];
        NSRectFill(NSMakeRect(0, 0, 240, 144));
        NSMutableParagraphStyle *style = [NSMutableParagraphStyle new];
        style.lineSpacing = 2;
        style.lineBreakMode = NSLineBreakByWordWrapping;
        NSDictionary *attrs = @{
            NSFontAttributeName : [NSFont systemFontOfSize:12],
            NSForegroundColorAttributeName : [NSColor whiteColor],
            NSParagraphStyleAttributeName : style
        };
        NSAttributedString *a = [[NSAttributedString alloc] initWithString:text attributes:attrs];
        NSRect bounds = [a boundingRectWithSize:NSMakeSize(224, 10000)
                                        options:NSStringDrawingUsesLineFragmentOrigin |
                                                NSStringDrawingUsesFontLeading];
        CGFloat h = ceil(bounds.size.height);
        [a drawWithRect:NSMakeRect(8, MAX(0, 144 - h), 224, h)
                options:NSStringDrawingUsesLineFragmentOrigin | NSStringDrawingUsesFontLeading];
        [NSGraphicsContext restoreGraphicsState];
        unsigned char *data = rep.bitmapData;
        unsigned char out[17280] = {0};
        for (int y = 0; y < 144; y++)
            for (int x = 0; x < 240; x++)
                out[y * 120 + x / 2] |= (data[y * 960 + x * 4] >> 4) << (x % 2 ? 0 : 4);
        if (argc > 2) {
            NSData *png = [rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
            [png writeToFile:[NSString stringWithUTF8String:argv[2]] atomically:YES];
        }
        fwrite(out, 1, sizeof(out), stdout);
        return 0;
    }
}
